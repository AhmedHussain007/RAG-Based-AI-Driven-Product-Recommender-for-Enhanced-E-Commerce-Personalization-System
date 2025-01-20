"""GenAI URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from GenAI import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin-page/', admin.site.urls),
    path('' , views.HomePage , name='home'),
    path('products/' , views.ProductsPage, name='products-page'),
    path('sign-up/' , views.SignupPage, name='sign-up-page'),
    path('log-in/' , views.LoginPage, name='log-in-page'),
    path('product/<int:id>/' , views.ProductPage , name = 'product-page'),
    path('contact-us/' , views.Contact_us , name = 'contact-us'),
    path('about-us/' , views.About_us , name = 'about-us'),
    path('cart/' , views.Cart , name = 'cart')
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
