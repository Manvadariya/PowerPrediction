import random
import string
from django import forms
from django.core.mail import send_mail
from .models import Userstable,Roles,UserRole
from django.conf import settings
from django.core.validators import validate_email
from django.core.exceptions import ValidationError

class LoginForm(forms.Form):
    username = forms.CharField(label='Username', max_length=50)
    password = forms.CharField(label='Password', widget=forms.PasswordInput)

class UserRegistrationForm(forms.ModelForm):
    role = forms.ModelChoiceField(
        queryset=Roles.objects.exclude(role_names__in=['Admin', 'Government Engineer']),
        required=True,
        label="Select Role"
    )

    class Meta:
        model = Userstable
        fields = ['username', 'password']
        widgets = {
            'password': forms.PasswordInput(),
        }

    def save(self, commit=True):
        user = super().save(commit=False)
        if commit:
            user.save()
            role = self.cleaned_data['role']
            UserRole.objects.create(user=user, role=role)
        return user


def generate_random_password(length=8):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


import socket
from django.core.validators import validate_email
from django.core.exceptions import ValidationError

class AdminUserCreationForm(forms.ModelForm):
    role = forms.ModelChoiceField(
        queryset=Roles.objects.filter(role_names__in=['Admin', 'Government Engineer']),
        required=True,
        label="Select Role"
    )

    class Meta:
        model = Userstable
        fields = ['username']

    def clean_username(self):
        username = self.cleaned_data.get('username')

        if username:
            try:
                validate_email(username)
            except ValidationError:
                raise forms.ValidationError("Please enter a valid email address as username for Admin or Government Engineer.")

            if Userstable.objects.filter(username=username).exists():
                raise forms.ValidationError("This email is already taken. Please choose a different one.")

            domain = username.split('@')[-1]
            try:
                socket.gethostbyname(domain)
            except socket.gaierror:
                raise forms.ValidationError("Invalid email domain. Please enter a valid email address.")

        return username
    
    def save(self, commit=True):
        user = super().save(commit=False)
        password = generate_random_password()
        user.password = password

        if commit:
            user.save()
            role = self.cleaned_data['role']
            UserRole.objects.create(user=user, role=role)
            
            send_mail(
                subject="Your Account Credentials",
                message=f"Hello {user.username},\n\nYour account has been created.\nUsername: {user.username}\nPassword: {password}\n\n",
                from_email=settings.EMAIL_HOST_USER,
                recipient_list=[user.username],
                fail_silently=False,
            )

        return user

class PasswordChangeForm(forms.Form):
    username = forms.CharField(max_length=50, label='User Email (Username)', widget=forms.TextInput(attrs={'class': 'form-control'}))

class DeleteCredentialForm(forms.Form):
    username = forms.CharField(
        label="Enter Username (Admin or Government Engineer)",
        max_length=50,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'})
    )

    def clean_username(self):
        username = self.cleaned_data['username']
        try:
            user = Userstable.objects.get(username=username)

            try:
                user_role = UserRole.objects.get(user=user).role.role_names
            except UserRole.DoesNotExist:
                raise forms.ValidationError("This user has no assigned role and cannot be deleted.")

            if user_role not in ['Admin', 'Government Engineer']:
                raise forms.ValidationError("You can only delete Admin or Government Engineer accounts.")

        except Userstable.DoesNotExist:
            raise forms.ValidationError("User does not exist.")
        
        return username

class NormalUserCredentialUpdateForm(forms.ModelForm):
    new_username = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'New Username'})
    )
    new_password = forms.CharField(
        max_length=128,
        required=True,
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'New Password'})
    )

    class Meta:
        model = Userstable
        fields = []

    def clean(self):
        cleaned_data = super().clean()
        new_username = cleaned_data.get('new_username')
        new_password = cleaned_data.get('new_password')

        if not new_username or not new_password:
            raise forms.ValidationError("Both username and password must be provided.")

        if Userstable.objects.filter(username=new_username).exists():
            self.add_error('new_username', "Username already exists. Please choose a different one.")

        return cleaned_data    
