from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import *
from btp_melanoma_app import views
  
urlpatterns = [
    # path('', views.home, name='home'),
    path('skinimage', predict_view, name = 'skin_image'),
]