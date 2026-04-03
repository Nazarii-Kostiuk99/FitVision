from django.contrib import admin
from django.urls import path, include, re_path
from api.views import serve_media

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
    re_path(r'^media/(?P<path>.*)$', serve_media),
]
