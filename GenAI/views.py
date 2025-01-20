from django.http import HttpResponse
from django.shortcuts import render,redirect
from products.models import Product,UserProductClick
from django.core.paginator import Paginator
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate,logout
from django.contrib.auth import login as auth_login
from . import first_order_logic
import random

def HomePage(request):
    indices_1 = random.sample(range(1, 51), 10)  # List of 20 random numbers from 1 to 50
    indices_2 = random.sample(range(1, 51), 10)
    if request.user.is_authenticated:
        print("User : " , request.user.id)
        recommended_products_1 = first_order_logic.recommend_products(request.user.id)
        from GenAI.genai import get_recommendation_using_gemini,manual
        recommended_products_2 = get_recommendation_using_gemini(request.user.id)        
        if recommended_products_1:
            indices_1[:10] = recommended_products_1
        if recommended_products_2 is not None:
            indices_2[:10] = recommended_products_2
        manual(request.user.id)
    products_1 = Product.objects.filter(id__in=indices_1)
    products_2 = Product.objects.filter(id__in=indices_2)
    products_1 = sorted(products_1, key=lambda p: indices_1.index(p.id))
    products_2 = sorted(products_2, key=lambda p: indices_2.index(p.id))
    data = {'products1':products_1 ,'products2':products_2}
    
    return render(request , "ecommerce.html" , data)

def ProductsPage(request):
    products = Product.objects.all().order_by('name')
    paginator = Paginator(products , 15)
    page_number = request.GET.get('page')
    products = paginator.get_page(page_number)
    prev_page = products.number - 1 if products.number > 1 else None
    next_page = products.number + 1 if products.number < paginator.num_pages else None
    data = {'products':products, "prev" : prev_page , "next": next_page}
    if request.user.is_authenticated:
        first_order_logic.recommend_products(request.user.id)
        print("logged in")
    return render(request,"products_page.html" , data)

def ProductPage(request, id):
    try:
        product = Product.objects.get(id=id)
        if request.user.is_authenticated:
            UserProductClick.objects.create(user=request.user, product=product)
            print("New click Added")
    except Product.DoesNotExist:
        return render(request, '404.html')
    return render(request, 'product_page.html', {'product': product})


def SignupPage(request):
    if (request.method == 'POST'):
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm-password')
        if(password != confirm_password):
            messages.error(request , "Passwords do not match")
            return render(request , 'sign_up.html')
        else:
            try:
                if User.objects.filter(username=username).exists():
                    messages.error(request, "Username already exists")
                    return render(request, 'sign_up.html')
                else:
                    user = User.objects.create_user(username=username , password=password)
                    auth_login(request , user)
                    return redirect('home')
            except:
                messages.error(request, "Error in creating user")
    return render(request,'sign_up.html')

def LoginPage(request):
    # Iterate through all users and print their details (for debugging only)
    users = User.objects.all()
    for user in users:
        print(f"Username: {user.username}")
        print(f"First Name: {user.first_name}")
        print(f"Last Name: {user.last_name}")
        print(f"Email: {user.email}")
        print(f"Is Active: {user.is_active}")
        print(f"Is Staff: {user.is_staff}")
        print(f"Is Superuser: {user.is_superuser}")
        print(f"Date Joined: {user.date_joined}")
        print("-" * 50)

    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # Check if user exists
        try:
            user = User.objects.get(username=username)
            print("Found User:", user.username)  # Debugging line to check if user is found
        except User.DoesNotExist:  # Catch only DoesNotExist exceptions
            messages.error(request, f"The {username} you entered does not exist.")
            return redirect('log-in-page')  # Redirect back to login page

        # Authenticate the user
        user_authenticated = authenticate(request=request, username=username, password=password)
        if user_authenticated is not None:
            auth_login(request, user_authenticated)  # Log the authenticated user in
            return redirect('home')  # Redirect to home page if authenticated successfully
        else:
            messages.error(request, "Invalid username or password.")  # Authentication failed
            return redirect('log-in-page')  # Redirect back to login page for another attempt

    return render(request, 'login.html')

def Contact_us(request):
    return render(request,'contact-us.html')

def About_us(request):
    return render(request,'about-us.html')

def Cart(request):
    return render(request,'cart.html')


