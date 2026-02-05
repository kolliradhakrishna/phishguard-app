import os
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'PhishingDetection.settings')
django.setup()

from django.contrib.auth import get_user_model

def create_superuser():
    User = get_user_model()
    username = os.environ.get('DJANGO_SUPERUSER_USERNAME', 'admin')
    email = os.environ.get('DJANGO_SUPERUSER_EMAIL', 'admin@example.com')
    password = os.environ.get('DJANGO_SUPERUSER_PASSWORD', 'admin123')

    if User.objects.filter(username=username).exists():
        print(f"Superuser '{username}' already exists.")
    else:
        User.objects.create_superuser(username, email, password)
        print(f"Superuser '{username}' created successfully.")

if __name__ == '__main__':
    create_superuser()
