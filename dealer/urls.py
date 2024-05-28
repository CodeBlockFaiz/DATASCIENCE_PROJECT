from django.urls import path 
from.import views

urlpatterns = [
    path('',views.index,name='index'),
    path('index',views.index,name='Home'),
    path('buy',views.buy,name='buy'),
    path('Sell',views.Sell,name='Sell'),
   path('contact',views.contact,name='contact'),
   path('register',views.register,name='register'),
   path('login',views.login,name='login'),
   path('logout',views.logout,name='logout'),
   path('otp',views.otpVerify,name='otpVerify'),
   path('predict',views.predict,name='predict'),
    path('carpredict',views.carpredict,name='carpredict')
]
