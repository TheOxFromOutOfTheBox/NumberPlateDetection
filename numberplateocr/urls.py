from django.urls import path,include
from . import views

urlpatterns=[
    path("index/",views.index,name="ocr-index"),
    path("imgupload/",views.imgupload,name="ocr-imgupload"),
    path("imgupload1/",views.ImageUploadAPI.as_view(),name="ocr-imgupload1"),
]