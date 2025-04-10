import faiss  # For vector database
import shap
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.ensemble import RandomForestRegressor
from GenAI import first_order_logic 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from products.models import Product as ModelProduct,UserProductClick
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer


genai.configure(api_key="AIzaSyAyv64aLWQw6wvugGGduj99XPoQl5-N33g")
model = genai.GenerativeModel("gemini-2.0-flash-exp")
def load_data():
    products = ModelProduct.objects.all()
    df = pd.DataFrame([{
    'id': product.id,
    'name': product.name,
    'min_price': product.min_price,
    'max_price': product.max_price,
    'description': product.description,
    'categories': product.categories.split(','),
    'tags': product.tags_list.split(',')
    } for product in products])
    return df



def load_user_clicks(user_id):
    user_clicks = UserProductClick.objects.filter(user__id = user_id)
    user_clicks_df = pd.DataFrame([{
        'user_id': click.user.id,
        'product_id':click.product.id
    } for click in user_clicks])
    return user_clicks_df

def get_recommendation_using_gemini(user_id):
    df = load_data()
    clicks = load_user_clicks(user_id)
    response = model.generate_content(f"Here is the list of products that are present in my data base {df} and here are the products clicked by the user{clicks} so recommend me the 10 best products that are same as clicked products note only give me the product id's in the form of a string where id's separated by commas like '6,9,23,24,20,21,22,25,2,31\n'don't give anything else in response")
    try:
        s = response.text[:-2]
        lst = [int(x.strip()) for x in s.split(',')]
        if len(clicks) != 0:
            print("Gemini Recommendation" , lst)
            referred = df[df['id'].isin(lst)].copy()
            # explain_recommendations(clicks, referred)
            return lst
        else:
            print("Returing None")
            return None
    except:
        print("Error occured in llm.")
        if clicks is not None and not clicks.empty:
            lst = first_order_logic.recommend_products(user_id)
            referred = df[df['id'].isin(lst)].copy()
            # recommend_and_explain(clicks, referred)
            print("In LLM reffered by FOL " , referred)
            return lst        

def load_data():
    products = ModelProduct.objects.all()
    df = pd.DataFrame([{
    'id': product.id,
    'name': product.name,
    'min_price': product.min_price,
    'max_price': product.max_price,
    'description': product.description,
    'categories': product.categories.split(','),
    'tags': product.tags_list.split(',')
    } for product in products])
    return df

def convert_to_vectors(df):
    """
    Converts the input DataFrame into a combined vector of text and numeric features.
    """
    # Combine text columns into a single column
    df['combined_text'] = df['name'] + ' ' + df['description'] + ' ' + df['categories'].apply(lambda x: ' '.join(x)) + ' ' + df['tags'].apply(lambda x: ' '.join(x))
    # Create the TF-IDF pipeline for the text data
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=100))
    ])

    # Create the pipeline for numeric data
    numeric_pipeline = Pipeline([
        ('scaler', MinMaxScaler())
    ])

    # Combine both using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_pipeline, 'combined_text'), 
            ('numeric', numeric_pipeline, ['min_price', 'max_price'])
        ]
    )

    # Ensure numeric data is clean
    df['min_price'] = pd.to_numeric(df['min_price'], errors='coerce').fillna(0)
    df['max_price'] = pd.to_numeric(df['max_price'], errors='coerce').fillna(0)

    # Transform data to vectors
    product_vectors = preprocessor.fit_transform(df)
        
    return product_vectors , preprocessor

def store_in_vector_db():
    df = load_data()  # Load the data
    vectors , _ = convert_to_vectors(df)  # Assuming this returns a csr_matrix
    product_ids = df['id'].values
    
    # Convert sparse matrix to dense format (if needed)
    if hasattr(vectors, 'toarray'):
        vectors = vectors.toarray()  # Convert CSR matrix to dense numpy array
    
    # Check if vectors are float32 (FAISS requires this)
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    # Check the shape and reshape if needed
    if len(vectors.shape) != 2:
        vectors = vectors.reshape(vectors.shape[0], -1)
    
    # Create an index (assumes vectors have fixed dimension d)
    d = vectors.shape[1]  # Dimension of each vector
    index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)

    # Train the index (necessary for some FAISS indices, like IVF)
    try:
        index.train(vectors)  # Only needed if you're using an index that requires training (e.g., IVF)
        index.add(vectors)
    except ValueError as e:
        print(f"Error adding vectors to FAISS index: {e}")
        print(f"Shape of vectors: {vectors.shape}")
        print(f"Data type of vectors: {vectors.dtype}")
    
    # Store product IDs as metadata for later lookup
    id_map = {i: product_id for i, product_id in enumerate(product_ids)}
    print("Vectors Stored in database.")
    return index, id_map


def recommend_similar_products(user_liked_vectors, index, id_map, top_n=5):
    # Ensure user_liked_vectors is 2D and in float32 format
    if len(user_liked_vectors.shape) != 2:
        # print(f"Warning: Reshaping vectors from shape {user_liked_vectors.shape} to 2D.")
        user_liked_vectors = user_liked_vectors.reshape(user_liked_vectors.shape[0], -1)
    
    if user_liked_vectors.dtype != np.float32:
        user_liked_vectors = user_liked_vectors.astype(np.float32)
    
    try:
        # Perform the search for similar products
        _, similar_indices = index.search(user_liked_vectors, top_n)
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

    # Flatten the indices and map them to product IDs
    similar_product_ids = set()
    for indices in similar_indices:
        for idx in indices:
            # Make sure the index is valid and exists in the id_map
            if idx in id_map:
                similar_product_ids.add(id_map[idx])
    # Return the list of similar product IDs (if any)
    return (similar_product_ids)

def explain_recommendations_with_shap(user_liked_df, recommended_df):
    # Combine user liked and recommended data for SHAP analysis
    combined_df = pd.concat([user_liked_df, recommended_df])
    labels = [0] * len(user_liked_df) + [1] * len(recommended_df)

    # Generate feature vectors
    vectors, preprocessor = convert_to_vectors(combined_df)

    # Train a simple model
    model = RandomForestRegressor()
    model.fit(vectors, labels)

    # Create SHAP explainer
    explainer = shap.Explainer(model, vectors)
    shap_values = explainer(vectors)

    # Generate explanations in natural language
    explanations = []
    for i, recommended_product in recommended_df.iterrows():
        # Find the clicked product(s) most similar to this recommended product
        recommended_vector = vectors[len(user_liked_df) + i]
        similarities = []

        for j, user_liked_product in user_liked_df.iterrows():
            user_vector = vectors[j]
            similarity = np.dot(recommended_vector, user_vector) / (
                np.linalg.norm(recommended_vector) * np.linalg.norm(user_vector)
            )
            similarities.append((j, similarity))

        # Sort clicked products by similarity
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        most_similar_idx = similarities[0][0] if similarities else None

        # Get the top clicked product name if available
        if most_similar_idx is not None:
            influencing_product = user_liked_df.iloc[most_similar_idx]
            influencing_product_name = influencing_product['name']
        else:
            influencing_product_name = "unknown"

        # Analyze SHAP values for the recommended product
        feature_contributions = shap_values[len(user_liked_df) + i].values
        feature_names = preprocessor.get_feature_names_out()

        top_features = sorted(
            zip(feature_names, feature_contributions),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]  # Top 3 contributing features

        explanation = f"Product '{recommended_product['name']}' was recommended because of its similarity to the clicked product '{influencing_product_name}'. "
        explanation += "Top contributing features were: "
        explanation += ', '.join(
            [f"'{feature}' contributed {round(contribution, 2)}" for feature, contribution in top_features]
        )
        explanations.append(explanation)

    return explanations



# Example usage
# Assuming `user_liked_products` is a DataFrame of products liked by the user (in the same format as `df` earlier)
def manual(user_id):
    index, id_map = store_in_vector_db()
    user_liked_products = load_user_clicks(user_id)
    prod_ids = user_liked_products['product_id'].astype(int).tolist()
    user_clicked_products = ModelProduct.objects.filter(id__in = prod_ids)
    click_df = pd.DataFrame([{
    'id': product.id,
    'name': product.name,
    'min_price': product.min_price,
    'max_price': product.max_price,
    'description': product.description,
    'categories': product.categories.split(','),
    'tags': product.tags_list.split(',')
    } for product in user_clicked_products])
    user_liked_vectors,_ = convert_to_vectors(click_df)
    recommended_product_ids = recommend_similar_products(user_liked_vectors, index, id_map, top_n=5)

    # Fetch product details from your database
    recommended_products = ModelProduct.objects.filter(id__in=recommended_product_ids)

    # Extract product names from the Product objects
    product_ids = [product.id for product in recommended_products]
    user_recommended_products = ModelProduct.objects.filter(id__in = prod_ids)
    recommended_df = pd.DataFrame([{
    'id': product.id,
    'name': product.name,
    'min_price': product.min_price,
    'max_price': product.max_price,
    'description': product.description,
    'categories': product.categories.split(','),
    'tags': product.tags_list.split(',')
    } for product in user_recommended_products])
    print("Products Recommended Using DB : " , product_ids[:10])
    explanations = explain_recommendations_with_shap(click_df, recommended_df)
    print("Starting Explanation\n")
    for exp in explanations:
        print(exp)



