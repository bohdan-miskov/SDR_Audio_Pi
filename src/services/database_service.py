import math
from typing import List, Any, Optional, Callable

from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool, pyqtSlot
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    JSON,
    ForeignKey,
    func,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    sessionmaker,
    declarative_base,
    relationship,
    joinedload,
    Session,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.event import listen

from src.models.detection_object import DetectionObject
from src.models.object_class import ObjectClass
from src.models.service_response import ServiceResponse, StatusCode, DbOperation

DB_CONNECTION_STRING: str = "sqlite:///./sdr_pi.db"

Base: Any = declarative_base()


class ObjectClassEntity(Base):
    __tablename__ = "object_classes"
    id: int = Column(Integer, primary_key=True, autoincrement=True)
    name: str = Column(String, unique=True, nullable=False)
    signatures = relationship("Signature", back_populates="object_class_rel")


class Signature(Base):
    __tablename__ = "signatures"
    id: int = Column(Integer, primary_key=True, autoincrement=True)
    name: str = Column(String, nullable=False)
    class_id: int = Column(
        Integer, ForeignKey("object_classes.id"), nullable=False, index=True
    )
    is_dangerous: bool = Column(Boolean, default=False)
    rf_params: Optional[List[str]] = Column(JSON, nullable=True)
    sound_params: Optional[List[float]] = Column(JSON, nullable=True)
    object_class_rel = relationship("ObjectClassEntity", back_populates="signatures")


class DBWorker(QRunnable):
    def __init__(self, func: Callable[..., None], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.func(*self.args, **self.kwargs)
        except Exception as e:
            print(f"[DB Worker Error] {e}")


class DatabaseService(QObject):
    # Єдиний сигнал для результату операцій
    request_finished = pyqtSignal(ServiceResponse)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.threadpool: QThreadPool = QThreadPool()
        self.engine: Engine = create_engine(
            DB_CONNECTION_STRING,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        listen(self.engine, "connect", self._enable_wal)
        Base.metadata.create_all(self.engine)
        self.Session: sessionmaker[Session] = sessionmaker(bind=self.engine)
        print(f"[DB] Service started. Mode: WAL enabled.")

    @staticmethod
    def _enable_wal(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    # --- Public Methods ---

    def request_objects_page(self, page: int = 1, page_size: int = 10) -> None:
        worker = DBWorker(self._fetch_page_task, page, page_size)
        self.threadpool.start(worker)

    def request_all_objects(self) -> None:
        worker = DBWorker(self._fetch_all_task)
        self.threadpool.start(worker)

    def add_object(self, obj_data: DetectionObject) -> None:
        worker = DBWorker(self._add_object_task, obj_data)
        self.threadpool.start(worker)

    def update_object(self, obj_data: DetectionObject) -> None:
        worker = DBWorker(self._update_object_task, obj_data)
        self.threadpool.start(worker)

    def delete_object(self, object_id: int) -> None:
        worker = DBWorker(self._delete_object_task, object_id)
        self.threadpool.start(worker)

    def request_classes(self) -> None:
        worker = DBWorker(self._fetch_classes_task)
        self.threadpool.start(worker)

    def add_class(self, class_data: ObjectClass) -> None:
        worker = DBWorker(self._add_class_task, class_data)
        self.threadpool.start(worker)

    def update_class(self, class_data: ObjectClass) -> None:
        worker = DBWorker(self._update_class_task, class_data)
        self.threadpool.start(worker)

    def delete_class(self, class_id: int) -> None:
        worker = DBWorker(self._delete_class_task, class_id)
        self.threadpool.start(worker)

    # --- Internal Tasks ---
    def _signature_to_dto(self, s: Signature) -> DetectionObject:
        return DetectionObject(
            id=s.id,
            name=str(s.name),
            class_id=s.class_id,
            object_class=s.object_class_rel.name if s.object_class_rel else "Unknown",
            is_dangerous=bool(s.is_dangerous),
            rf_params_hz=s.rf_params or [],
            sound_params_hz=s.sound_params or [],
        )

    def _fetch_all_task(self) -> None:
        session: Session = self.Session()

        resp = ServiceResponse(
            operation=DbOperation.GET_ALL_OBJECTS,
            status=StatusCode.INTERNAL_ERROR,
            message="Init",
        )

        try:
            signatures = (
                session.query(Signature)
                .options(joinedload(Signature.object_class_rel))
                .order_by(Signature.id.desc())
                .all()
            )

            items_data = [self._signature_to_dto(s).to_dict() for s in signatures]
            count = len(items_data)

            resp.status = StatusCode.OK
            resp.message = f"Successfully loaded {count} objects"

            resp.data = {"items": items_data, "count": count}

        except Exception as e:
            print(f"[DB Error Fetch All] {e}")
            resp.message = f"Error fetching all objects: {str(e)}"

        finally:
            session.close()
            self.request_finished.emit(resp)

    def _fetch_page_task(self, page: int, page_size: int) -> None:
        session: Session = self.Session()
        resp = ServiceResponse(
            operation=DbOperation.GET_OBJECTS_PAGE,
            status=StatusCode.INTERNAL_ERROR,
            message="Init",
        )

        try:
            total_items: int = session.query(func.count(Signature.id)).scalar() or 0
            total_pages: int = math.ceil(total_items / page_size) if page_size else 0
            page = max(1, min(page, total_pages)) if total_pages > 0 else 1
            offset: int = (page - 1) * page_size

            signatures = (
                session.query(Signature)
                .options(joinedload(Signature.object_class_rel))
                .order_by(Signature.id.desc())
                .limit(page_size)
                .offset(offset)
                .all()
            )

            items_data = [self._signature_to_dto(s).to_dict() for s in signatures]

            resp.status = StatusCode.OK
            resp.message = "Page loaded"

            resp.data = {
                "items": items_data,
                "page": page,
                "total": total_items,
                "total_pages": total_pages,
            }

        except Exception as e:
            resp.message = str(e)
        finally:
            session.close()
            self.request_finished.emit(resp)

    def _fetch_classes_task(self) -> None:
        session: Session = self.Session()
        resp = ServiceResponse(
            operation=DbOperation.GET_CLASSES,
            status=StatusCode.INTERNAL_ERROR,
            message="Init",
        )

        try:
            results = session.query(ObjectClassEntity).all()
            classes_dicts = [
                ObjectClass(id=row.id, name=str(row.name)).to_dict() for row in results
            ]

            resp.status = StatusCode.OK
            resp.message = "Classes loaded"
            resp.data = {"classes": classes_dicts}

        except Exception as e:
            resp.message = str(e)
        finally:
            session.close()
            self.request_finished.emit(resp)

    # --- UPDATED CRUD TASKS (Logic with Status Codes) ---

    def _add_object_task(self, obj_data: DetectionObject) -> None:
        session: Session = self.Session()
        resp = ServiceResponse(
            operation=DbOperation.ADD_OBJECT,
            status=StatusCode.INTERNAL_ERROR,
            message="Init",
        )

        try:
            if not obj_data.name:
                raise ValueError("Object name is required")

            target_class_id = obj_data.class_id
            target_class_name = obj_data.object_class

            # Логіка пошуку класу...
            if not target_class_id:
                obj_class = (
                    session.query(ObjectClassEntity)
                    .filter_by(name=obj_data.object_class)
                    .first()
                )
                if not obj_class:
                    resp.status = StatusCode.BAD_REQUEST
                    resp.message = f"Class '{obj_data.object_class}' not found"
                    return
                target_class_id = obj_class.id
                target_class_name = obj_class.name
            else:
                obj_class = session.query(ObjectClassEntity).get(target_class_id)
                if obj_class:
                    target_class_name = obj_class.name

            new_sig = Signature(
                name=obj_data.name,
                class_id=target_class_id,
                is_dangerous=obj_data.is_dangerous,
                rf_params=obj_data.rf_params_hz,
                sound_params=obj_data.sound_params_hz,
            )

            session.add(new_sig)
            session.commit()
            session.refresh(new_sig)

            created_dto = DetectionObject(
                id=new_sig.id,
                name=new_sig.name,
                class_id=new_sig.class_id,
                object_class=target_class_name,
                is_dangerous=new_sig.is_dangerous,
                rf_params_hz=new_sig.rf_params,
                sound_params_hz=new_sig.sound_params,
            )

            resp.status = StatusCode.CREATED
            resp.message = "Object successfully added"
            resp.data = created_dto.to_dict()

        except ValueError as e:
            session.rollback()
            resp.status = StatusCode.BAD_REQUEST
            resp.message = str(e)
        except IntegrityError:
            session.rollback()
            resp.status = StatusCode.CONFLICT
            resp.message = "Database integrity error (duplicate?)"
        except Exception as e:
            session.rollback()
            resp.status = StatusCode.INTERNAL_ERROR
            resp.message = f"DB Error: {str(e)}"
        finally:
            session.close()
            self.request_finished.emit(resp)

    def _update_object_task(self, obj_data: DetectionObject) -> None:
        session: Session = self.Session()
        resp = ServiceResponse(
            operation=DbOperation.UPDATE_OBJECT,
            status=StatusCode.INTERNAL_ERROR,
            message="Init",
        )

        try:
            if obj_data.id is None:
                raise ValueError("ID required")

            sig = session.query(Signature).get(obj_data.id)
            if not sig:
                resp.status = StatusCode.NOT_FOUND
                resp.message = f"Object with ID {obj_data.id} not found"
                return

            target_class_id = obj_data.class_id or sig.class_id

            sig.name = obj_data.name
            sig.class_id = target_class_id
            sig.is_dangerous = obj_data.is_dangerous
            sig.rf_params = obj_data.rf_params_hz
            sig.sound_params = obj_data.sound_params_hz

            session.commit()
            session.refresh(sig)

            updated_dto = self._signature_to_dto(sig)

            resp.status = StatusCode.OK
            resp.message = "Object updated"
            resp.data = updated_dto.to_dict()

        except Exception as e:
            session.rollback()
            resp.status = StatusCode.INTERNAL_ERROR
            resp.message = str(e)
        finally:
            session.close()
            self.request_finished.emit(resp)

    def _delete_object_task(self, object_id: int) -> None:
        session: Session = self.Session()
        resp = ServiceResponse(
            operation=DbOperation.DELETE_OBJECT,
            status=StatusCode.INTERNAL_ERROR,
            message="Init",
        )

        try:
            rows = session.query(Signature).filter(Signature.id == object_id).delete()
            session.commit()

            if rows == 0:
                resp.status = StatusCode.NOT_FOUND
                resp.message = "Object not found"
            else:
                resp.status = StatusCode.OK
                resp.message = "Object deleted"
                resp.data = {"id": object_id}

        except Exception as e:
            session.rollback()
            resp.status = StatusCode.INTERNAL_ERROR
            resp.message = str(e)
        finally:
            session.close()
            self.request_finished.emit(resp)

    def _add_class_task(self, class_data: ObjectClass) -> None:
        session: Session = self.Session()
        resp = ServiceResponse(
            operation=DbOperation.ADD_CLASS,
            status=StatusCode.INTERNAL_ERROR,
            message="Init",
        )

        try:
            if not class_data.name:
                raise ValueError("Class name required")

            existing = (
                session.query(ObjectClassEntity).filter_by(name=class_data.name).first()
            )
            if existing:
                resp.status = StatusCode.CONFLICT
                resp.message = f"Class '{class_data.name}' already exists"
                return

            new_class = ObjectClassEntity(name=class_data.name)
            session.add(new_class)
            session.commit()
            session.refresh(new_class)

            resp.status = StatusCode.CREATED
            resp.message = "Class added"
            resp.data = ObjectClass(id=new_class.id, name=new_class.name).to_dict()

        except ValueError as e:
            resp.status = StatusCode.BAD_REQUEST
            resp.message = str(e)
        except Exception as e:
            session.rollback()
            resp.status = StatusCode.INTERNAL_ERROR
            resp.message = str(e)
        finally:
            session.close()
            self.request_finished.emit(resp)

    def _update_class_task(self, class_data: ObjectClass) -> None:
        session: Session = self.Session()
        resp = ServiceResponse(
            operation=DbOperation.UPDATE_CLASS,
            status=StatusCode.INTERNAL_ERROR,
            message="Init",
        )

        try:
            if not class_data.id:
                raise ValueError("Class ID required")

            entity = session.query(ObjectClassEntity).get(class_data.id)
            if not entity:
                resp.status = StatusCode.NOT_FOUND
                resp.message = "Class not found"
                return

            # Check duplication if name changed
            if entity.name != class_data.name:
                existing = (
                    session.query(ObjectClassEntity)
                    .filter(ObjectClassEntity.name == class_data.name)
                    .first()
                )
                if existing:
                    resp.status = StatusCode.CONFLICT
                    resp.message = "Class name already taken"
                    return

            entity.name = class_data.name
            session.commit()
            session.refresh(entity)

            resp.status = StatusCode.OK
            resp.message = "Class updated"
            resp.data = ObjectClass(id=entity.id, name=entity.name).to_dict()

        except ValueError as e:
            resp.status = StatusCode.BAD_REQUEST
            resp.message = str(e)
        except Exception as e:
            session.rollback()
            resp.status = StatusCode.INTERNAL_ERROR
            resp.message = str(e)
        finally:
            session.close()
            self.request_finished.emit(resp)

    def _delete_class_task(self, class_id: int) -> None:
        session: Session = self.Session()
        resp = ServiceResponse(
            operation=DbOperation.DELETE_CLASS,
            status=StatusCode.INTERNAL_ERROR,
            message="Init",
        )

        try:
            usage = (
                session.query(func.count(Signature.id))
                .filter(Signature.class_id == class_id)
                .scalar()
            )
            if usage > 0:
                resp.status = StatusCode.CONFLICT
                resp.message = f"Cannot delete: Class used by {usage} objects"
                return

            rows = (
                session.query(ObjectClassEntity)
                .filter(ObjectClassEntity.id == class_id)
                .delete()
            )
            session.commit()

            if rows == 0:
                resp.status = StatusCode.NOT_FOUND
                resp.message = "Class not found"
            else:
                resp.status = StatusCode.OK
                resp.message = "Class deleted"
                resp.data = {"id": class_id}

        except Exception as e:
            session.rollback()
            resp.status = StatusCode.INTERNAL_ERROR
            resp.message = str(e)
        finally:
            session.close()
            self.request_finished.emit(resp)
