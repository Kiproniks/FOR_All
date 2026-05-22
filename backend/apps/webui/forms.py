from django import forms

from apps.accounts.models import User


class MultiFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class LoginForm(forms.Form):
    email = forms.EmailField(label="Email")
    password = forms.CharField(label="Password", widget=forms.PasswordInput)


class RegisterForm(forms.Form):
    email = forms.EmailField(label="Email")
    password1 = forms.CharField(label="Password", widget=forms.PasswordInput, min_length=8)
    password2 = forms.CharField(label="Repeat password", widget=forms.PasswordInput, min_length=8)

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("User with this email already exists.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        if cleaned_data.get("password1") != cleaned_data.get("password2"):
            self.add_error("password2", "Passwords do not match.")
        return cleaned_data


class UploadBooksForm(forms.Form):
    files = forms.FileField(
        label="Book files (FB2/PDF)",
        widget=MultiFileInput(attrs={"accept": ".fb2,.pdf,application/pdf"}),
    )


class ConceptEditForm(forms.Form):
    custom_explanation = forms.CharField(
        label="Custom explanation",
        widget=forms.Textarea(attrs={"rows": 4}),
        min_length=3,
    )
