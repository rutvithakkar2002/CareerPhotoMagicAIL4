from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.upload_view, name='home'),
    path('generated_images/<str:user_id>/', views.generated_images, name='generated_images'),
    path('download/<str:user_id>/<str:image_name>/', views.download_image, name='download_image'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
