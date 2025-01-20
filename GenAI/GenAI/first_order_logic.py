from  pyDatalog import pyDatalog as plog
import pandas as pd
from products.models import Product as ModelProduct,UserProductClick

plog.clear()
plog.create_terms('Product, HasPrice, User, HasCategory, HasTag, LikedByUser, user_id, prod_id, Recommended , Lies,upper_bound,lower_bound,P,SimilarCategory, SimilarTag , X , Y , Z , C , T')
def addFacts():
    products = ModelProduct.objects.all()
    for product in products:
        id = product.id
        name = product.name
        min_price = product.min_price
        categories = product.categories.split(',')
        tags = product.tags_list.split(',')

        +Product(id, name)
        +HasPrice(id, min_price)

        for category in categories:
            +HasCategory(id, category.strip())

        for tag in tags:
            +HasTag(id, tag.strip())

def addClickeds(id):
    userclickedprods = UserProductClick.objects.filter(user__id = id)  # Get all user-product click records from the database
    +LikedByUser(1 , 1)
    if userclickedprods:
        for i, click in enumerate(userclickedprods):
            user_id = click.user
            prod_num = click.product
            +LikedByUser(user_id.id, prod_num.id)

def intialize_logic():
    SimilarCategory(X , Y , Z) <= (LikedByUser(Z, X) & HasCategory(X, C) & HasCategory(Y, C) & (X != Y))
    SimilarTag(X, Y , Z) <= (LikedByUser(Z, X) & HasTag(X, T) & HasTag(Y, T) & (X != Y))
    Lies(X, P , lower_bound , upper_bound) <= HasPrice(X , P) & (P > lower_bound) & (P < upper_bound)

def get_rec_prods(clicked_products , user_id):
    product_similarity_dict = {}
    for product in clicked_products:
        prod_id = product[0]
        if isinstance(prod_id, tuple):
            prod_id = prod_id[0]
        similar_product = plog.ask(f'SimilarCategory({prod_id}, Y , {user_id})')
        if similar_product is not None:
            for p in similar_product.answers:
                product_id = p[0]
                if product_id in product_similarity_dict:
                    product_similarity_dict[product_id] += (len(HasCategory(prod_id, C) & HasCategory(product_id, C))*10)
                else:
                    product_similarity_dict[product_id] = (len(HasCategory(prod_id, C) & HasCategory(product_id, C))*10)
        
        similar_product = None
        similar_product = plog.ask(f'SimilarTag({prod_id}, Y , {user_id})')
        if similar_product is not None:
            for p in similar_product.answers:
                product_id = p[0]
                if product_id in product_similarity_dict:
                    product_similarity_dict[product_id] += (len(HasTag(prod_id, C) & HasTag(product_id, C))*5)
                else:
                    product_similarity_dict[product_id] = ((len(HasTag(prod_id, C) & HasTag(product_id, C))*5))
        
        # similar_product = None
        # base_price = HasPrice(1 , Y)[0][0]
        # lower_bound = float(base_price) * 0.8  # 20% less than base price
        # upper_bound = float(base_price) * 1.2  # 20% more than base price
        # similar_product = plog.ask(f'Lies(X, P , {lower_bound} , {upper_bound})')
        # if similar_product is not None:
        #     for p in similar_product.answers:
        #         product_id = p[0]
        #         if product_id in product_similarity_dict:
        #             product_similarity_dict[product_id] += 3
        #         else:
        #             product_similarity_dict[product_id] = 3
    
    return product_similarity_dict

def recommend_products(user_id, top_n=10):
    addFacts()
    addClickeds(user_id)
    intialize_logic()
    clicked_products = plog.ask(f'LikedByUser({user_id}, Y)')
    if clicked_products is None:
        print("No Clicked Products")
        return
    else:
        product_similarity_dict = get_rec_prods(clicked_products.answers , user_id)
        sorted_list = sorted(list(product_similarity_dict.items()), key=lambda x: x[1], reverse=True)
    recommended_products = [x[0] for x in sorted_list[:top_n]]
    print("Clicked : " , clicked_products)
    print("Recommended : " , recommended_products)
    explain_recommendations(clicked_products.answers , recommended_products, user_id)
    return recommended_products
    
def explain_recommendations(clicked_products , recommended_products , user_id):
    print("Explanation for choosing the recommended products")
    for clicked_product in clicked_products:
        clicked_product = clicked_product[0]
        ls = [(clicked_product , 4)]
        rec_prods = get_rec_prods(ls , user_id)
        for rec_prod in rec_prods:
            if rec_prod in recommended_products:
                explanation1 = (HasCategory(clicked_product, C) & HasCategory(rec_prod, C))
                explanation2 = (HasTag(clicked_product, C) & HasTag(rec_prod, C))
                samecats = []
                sametags = []
                for ex1 in explanation1:
                    samecats.append(ex1[0])
                for ex2 in explanation2:
                    sametags.append(ex2[0])
                if samecats and sametags:
                    print(f"Prod {rec_prod} recommended as prod {clicked_product} as it have categories {samecats} tags {sametags}")
                elif samecats:
                    print(f"Prod {rec_prod} recommended as prod {clicked_product} as it have categories {samecats}.")
                elif len(sametags) > 1:
                    print(f"Prod {rec_prod} recommended as prod {clicked_product} as it have tags {sametags}.")
                # else:
                #     print(f"Prod is recommended as it contain tags belong to multiple liked products")
    print("Fol Completed")