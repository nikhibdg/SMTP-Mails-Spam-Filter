from flask_wtf import Form
from wtforms import TextField, TextAreaField, SubmitField,validators,PasswordField
from wtforms.validators import Required

class ContactForm(Form):
    name = TextField("Name",  [validators.Required()])
    email = TextField("Email",  [validators.Required()])
    subject = TextField("Subject",  [validators.Required()])
    message = TextAreaField("Message",  [validators.Required()])
    submit = SubmitField("Send")

class SignupForm(Form):
    firstname = TextField("First name",  [validators.Required("Please enter your first name.")])
    lastname = TextField("Last name",  [validators.Required("Please enter your last name.")])
    email = TextField("Email",  [validators.Required("Please enter your email address."), validators.Email("Please enter your email address.")])
    password = PasswordField('Password', [validators.Required("Please enter a password.")])
    submit = SubmitField("Create account")

    def __init__(self, *args, **kwargs):
        Form.__init__(self, *args, **kwargs)

    def validate(self):
        if not Form.validate(self):
            return False

        user = User.query.filter_by(email = self.email.data.lower()).first()
        if user:
            self.email.errors.append("That email is already taken")
            return False
        else:
            return True
