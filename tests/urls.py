from django.conf.urls import url,include
from django.contrib import admin
from tests import views
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$',views.index),
    url(r'predict.html',views.predict),
    url()
 ];