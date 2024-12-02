from PyQt6 import QtWidgets, uic
import sys
import sqlite3
import subprocess
import sys
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QApplication, QDialog, QStackedWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QMessageBox, QLineEdit
from PyQt6.uic import loadUi
import sqlite3
from DS440_backend_final_v5 import train_and_generate_video

class Login(QDialog):
    def __init__(self):
        super(Login, self).__init__()
        loadUi(r"C:/Users/Ivan9/OneDrive/桌面/School/DS440/UI_files/login.ui", self)
        self.login_button.clicked.connect(self.loginfunction)
        self.pass_textbox.setEchoMode(QLineEdit.EchoMode.Password)
        self.createacc_button.clicked.connect(self.gotocreate)
        self.setFixedSize(480, 620)

    def loginfunction(self):
        username = self.user_textbox_1.text()
        password = self.pass_textbox.text()

        conn = sqlite3.connect(r"C:/Users/Ivan9/OneDrive/桌面/School/DS440/User_Credentials.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (username, password))
        result = cursor.fetchone()
        conn.close()

        if result:
            print("Successfully logged in with email:", username)
            while widget.count() > 1:
                widget.removeWidget(widget.widget(1))
                
            settings_screen = SettingsScreen()
            widget.addWidget(settings_screen)
            widget.setCurrentIndex(1)
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Login Failed")
            msg_box.setText("Invalid email or password")
            msg_box.exec()

    def gotocreate(self):
        createaccount = CreateAcc()
        widget.addWidget(createaccount)
        widget.setCurrentIndex(widget.currentIndex() + 1)


class CreateAcc(QDialog):
    def __init__(self):
        super(CreateAcc, self).__init__()
        loadUi(r"C:/Users/Ivan9/OneDrive/桌面/School/DS440/UI_files/signup.ui", self)
        self.signup_button.clicked.connect(self.createaccfunction)
        self.password_textbox.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirmpass_textbox.setEchoMode(QLineEdit.EchoMode.Password)
        self.backtologin.clicked.connect(self.gobacktologin)
        self.setFixedSize(480, 620)

    def createaccfunction(self):
        username = self.user_textbox_2.text()
        password = self.password_textbox.text()

        if password == self.confirmpass_textbox.text():
            try:
                conn = sqlite3.connect(r"C:/Users/Ivan9/OneDrive/桌面/School/DS440/User_Credentials.db")
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (username, password))
                conn.commit()
                conn.close()

                print("Successfully created account with email:", username)
                login = Login()
                widget.addWidget(login)
                widget.setCurrentIndex(widget.currentIndex() + 1)
            except sqlite3.IntegrityError:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Icon.Warning)
                msg_box.setWindowTitle("Account Creation Failed")
                msg_box.setText("An account with this email already exists.")
                msg_box.exec()
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Password Mismatch")
            msg_box.setText("Passwords do not match.")
            msg_box.exec()
    
    def gobacktologin(self):
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(0)  

class SettingsScreen(QDialog):
    def __init__(self):
        super(SettingsScreen, self).__init__()
        loadUi(r"C:/Users/Ivan9/OneDrive/桌面/School/DS440/UI_files/mainpage.ui", self)
        # Connect the simulation button
        self.simulation_button.clicked.connect(self.start_simulation)
        self.setFixedSize(580,632)

    def start_simulation(self):

        settings = {
            "shooting": self.comboBox_shoot.currentText(),
            "handles": self.comboBox_handles.currentText(),
            "passing": self.comboBox_pass.currentText()
        }
        print("Starting simulation with settings:", settings)
        train_and_generate_video(settings)

        # Show completion message
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Simulation Complete")
        msg_box.setText("The simulation has completed successfully. Check the generated video.")
        msg_box.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = Login()
    widget = QStackedWidget()
    widget.addWidget(main_window)
    widget.show()
    sys.exit(app.exec())
