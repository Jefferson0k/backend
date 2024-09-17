from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('lsp/', include('lsp.urls')),  # Incluye las rutas de la app 'lsp'
]