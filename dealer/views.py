from django.shortcuts import render,redirect
from django.contrib.auth.models import User,auth
from django.contrib import messages
import math
import random
from django.core.mail import send_mail
from django.http import HttpResponseRedirect

otp=None

def otpGen(request):
	string = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
	OTP = ''
	for i in range(6):
		OTP += string[math.floor(random.random() * len(string))]
	return OTP

def otpSend(request,user,otp):
    from redmail import outlook

    outlook.user_name = "syedfaizanmuyeez786@outlook.com"
    outlook.password = "Syed@2002"

    outlook.send(
            receivers=user.email,
            subject="OTP",
            #text="Hi, this is an example."
        text = """\
                Hi,
                 
                OTP : {0} 


                    \
                """.format(otp))
def otpVerify(request):
	global otp
	user = request.user
	if otp == None:
		otp = otpGen(request)
		otpSend(request,user,otp)
		return render(request,'otpform.html')
	else:
		if request.method == "POST":
			otpfield = request.POST['otp']
			if otpfield == otp:
				return redirect('/')
			else:
				auth.logout(request)
				return redirect('/')
		else:
			otp = otpGen(request)
			otpSend(request,user,otp)
			auth.logout(request)
			return render(request,'otpform.html')


# Create your views here.
def index(request):
  return render(request,"index.html")


def contact(request):
  return render(request,"contact.html")

def Sell(request):
  return render(request,"sell.html")

def buy(request):
  return render(request,"buy.html")

def register(request):
  if request.method=="POST":
        first=request.POST['FName']
        last=request.POST['LName']
        user=request.POST['Username']
        email=request.POST['email']
        p1=request.POST['psw']
        p2=request.POST['psw-repeat']
        if p1==p2:
            if User.objects.filter(username=user).exists():
                messages.info(request,"Username Exists")
                return render(request,"register.html")
            elif User.objects.filter(email=email).exists():
                messages.info(request,"Email Exists")
            else:
                u=User.objects.create_user(first_name=first,last_name=last,email=email,username=user,password=p2)
                u.save()
                return redirect('login')
        else:
            messages.info(request,"Password not matching")
            return render(request,"register.html")
  else:
        return render(request,"register.html")
  return render(request,"register.html")


def login(request):
   if request.method=="POST":
        u=request.POST['Username']
        p=request.POST['psw']
        user=auth.authenticate(username=u,password=p)
        print(u,p )
        if user is not None:
            auth.login(request,user)
            return HttpResponseRedirect('otp')
        else:
            messages.info(request,"Invalid Credentials")
            return render(request,"login.html")
   return render(request,"login.html")

def logout(request):
    auth.logout(request)
    return redirect("index")

def carpredict(request):
    return render(request,"carpredict.html")

def predict(request):
    if request.method=="POST":
        name=request.POST['name']
        year=float(request.POST['year'])
        km_driven=float(request.POST['km_driven'])
        fuel=request.POST['fuel']
        from sklearn.preprocessing import LabelEncoder
        l=LabelEncoder()
        l.fit_transform([name,fuel])

        n=l.fit_transform([name])
        f=l.fit_transform([fuel])
        
        import pandas as pd
        df=pd.read_csv(r'static/datasets/CAR DETAILS FROM CAR DEKHO.csv')
        print(df.head())
        print(df.isnull())
        print(df.isnull().sum())
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.heatmap(df.isnull())
        plt.show()
        n1=l.fit_transform(df['name'])
        f1=l.fit_transform(df['fuel'])

        df=df.drop(['name','fuel'],axis=1)
        df['name']=n1
        df['fuel']=f1

        x=df.drop(["seller_type","transmission","owner"],axis=1)
        Y=df["selling_price"]

        
        from sklearn.model_selection import train_test_split
        x_train,x_test,Y_train,Y_test=train_test_split(x,Y,test_size=0.30)
        
        plt.plot(x_test,Y_test)
        plt.show()
        from sklearn.linear_model import LinearRegression
        lin=LinearRegression()
        lin.fit(x_train,Y_train)
        predict=lin.predict(x_test) 
        from sklearn.tree import DecisionTreeRegressor
        tree=DecisionTreeRegressor()
        tree.fit(x_train,Y_train)
        predict_tree=tree.predict(x_test)
        
        plt.plot(x_test,predict,label="Linear Regression")
        plt.legend()
        plt.show()

        plt.plot(x_test,predict_tree,label="Decision Tree Regression")
        plt.legend()
        plt.show()

        from sklearn.metrics import mean_squared_error,r2_score
        print("Linear Regression")
        print("Mean Squared Error: ",mean_squared_error(predict,Y_test))
        print("R2 Score: ",r2_score(predict,Y_test))
        
        print("Decision Tree Regression")
        print("Mean Squared Error: ",mean_squared_error(predict_tree,Y_test))
        print("R2 Score: ",r2_score(predict_tree,Y_test))
        
        
        X_train=df[["name","year","km_driven","fuel"]]
        y_train=df["selling_price"]
        from sklearn.linear_model import LinearRegression
        reg=LinearRegression()
        reg.fit(X_train,y_train)
        import numpy as np
        pred=np.array([[n,year,km_driven,f]],dtype=object)
        predict_selling_price=int(reg.predict(pred))
        print(predict_selling_price)
    return render(request,"predict.html",{"name":name,"year":year,"km_driven":km_driven,"fuel":fuel,"selling_price":predict_selling_price})




# Create your views here.
