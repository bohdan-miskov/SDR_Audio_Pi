import sys
from PyQt6.QtCore import QCoreApplication
from src.services.pi_server_service import PiServerService

def main():
    # QCoreApplication потрібен, щоб Qt міг обробляти мережеві події
    app = QCoreApplication(sys.argv)
    
    # Ініціалізуємо сервіс
    server = PiServerService(port=6000)
    
    # Запускаємо сервер
    server.start()
    
    # Запускаємо цикл обробки подій Qt
    print("Сервер запущено. Натисніть Ctrl+C для зупинки.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()