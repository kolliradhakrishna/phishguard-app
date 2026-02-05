from django.urls import path

from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [path("", views.index, name="index"),
               path("index.html", views.index, name="index"),
               path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),	      
               path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
               path("RunLGBM", views.RunLGBM_Protected, name="RunLGBM"),
               path("RunSVM", views.RunSVM_Protected, name="RunSVM"),
               path("Predict", views.Predict, name="Predict"),
               path("PredictAction", views.PredictAction, name="PredictAction"),
               path("blockurl", views.blockurl, name="blockurl"),
               path("ViewBlockedUrls", views.ViewBlockedUrls,name="ViewBlockedUrls"),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
