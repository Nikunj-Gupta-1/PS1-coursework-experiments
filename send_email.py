import smtplib
# creates SMTP session
s = smtplib.SMTP('smtp.gmail.com', 587)
# start TLS for security
s.starttls()
# Authentication
s.login("hiitsforbluestacks@gmail.com", "Loverofthor01")
# message to be sent
message = "Test_message"
# sending the mail
s.sendmail("hiitsforbluestacks@gmail.com", "devanshi2006jain@gmail.com", message)
# terminating the session
s.quit()