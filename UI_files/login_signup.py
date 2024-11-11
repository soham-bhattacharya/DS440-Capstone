import sys
import subprocess
import sqlite3
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi


class Login(QDialog):
    def __init__(self):
        super(Login,self).__init__()
        loadUi("C:/Users/Ivan9/OneDrive/桌面/School/DS440/login.ui",self)
        self.login_button.clicked.connect(self.loginfunction)
        self.pass_textbox.setEchoMode(QtWidgets.QLineEdit.Password)
        self.createacc_button.clicked.connect(self.gotocreate)

    def loginfunction(self):
        email = self.email_textbox.text()
        password = self.pass_textbox.text()
        
        # Connect to the database to check credentials
        conn = sqlite3.connect("C:/Users/Ivan9/OneDrive/桌面/School/DS440/User_Credentials.db")
        cursor = conn.cursor()
        
        # Query to find a user with the entered email and password
        cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
        result = cursor.fetchone()
        conn.close()
        
        # Check if a matching record was found
        if result:
            print("Successfully logged in with email:", email)
            # Transition to the main application or next page
            #main_app = MainApp()
            #widget.addWidget(main_app)
            widget.setCurrentIndex(widget.currentIndex() + 1)
            try:
                subprocess.Popen(["python", r"C:/Users/Ivan9/OneDrive/桌面/School/DS440/DS440_basketball_reinforcement.py"])
            except Exception as e:
                print(f"Error launching backend: {e}")
        else:
            print("Invalid email or password")  # Display an error message in the console
            # Optionally, you can display a message box with an error
            msg_box = QtWidgets.QMessageBox()
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setWindowTitle("Login Failed")
            msg_box.setText("Invalid email or password")
            msg_box.exec_()



    def gotocreate(self):
        createaccount = CreateAcc()
        widget.addWidget(createaccount)
        widget.setCurrentIndex(widget.currentIndex() + 1)



class CreateAcc(QDialog):
    def __init__(self):
        super(CreateAcc, self).__init__()
        loadUi("C:/Users/Ivan9/OneDrive/桌面/School/DS440/signup.ui", self)
        self.signup_button.clicked.connect(self.createaccfunction)
        self.password_textbox.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirmpass_textbox.setEchoMode(QtWidgets.QLineEdit.Password)
        self.backtologin.clicked.connect(self.gobacktologin)

    def createaccfunction(self):
        email = self.email_textbox.text()  # Now using 'email_textbox' for email
        password = self.password_textbox.text()  # Now using 'password_textbox' for password
        
        # Ensure passwords match before attempting to store data
        if password == self.confirmpass_textbox.text():
            try:
                # Connect to the existing database file and insert new user data
                conn = sqlite3.connect("C:/Users/Ivan9/OneDrive/桌面/School/DS440/User_Credentials.db")
                cursor = conn.cursor()
                
                # Insert email and password into users table
                cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, password))
                conn.commit()
                conn.close()
                
                print("Successfully created account with email:", email)
                
                # Transition to the login page or main application
                login = Login()
                widget.addWidget(login)
                widget.setCurrentIndex(widget.currentIndex() + 1)

            except sqlite3.IntegrityError:
                print("An account with this email already exists.")
        else:
            print("Passwords do not match")

    def gobacktologin(self):
        # Navigate back to the login page
        login = Login()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex() - 1)  # Go back to the previous widget
    
app=QApplication(sys.argv)
mainwindow=Login()
widget=QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedWidth(480)
widget.setFixedHeight(620)
widget.show()
app.exec_()