from neo4j import GraphDatabase

class Neo4jManager:
    def __init__(self, uri, user, password):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()


    def create_product_node(self,product_id, name, min_price, max_price, description, categories, image, tags_list):
        with self.driver.session() as session:
            query = """
            CREATE (n:Product {
                id: $product_id,
                name: $name,
                min_price: $min_price,
                max_price: $max_price,
                description: $description,
                categories: $categories,
                image: $image,
                tags_list: $tags_list
            })
            """
            session.run(query,product_id=product_id, name=name, min_price=min_price, max_price=max_price, 
                        description=description, categories=categories, image=image, tags_list=tags_list)
            print(f"Product '{name}' created in Neo4j.")

# # Example usage:
# neo4j_conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", password="Ahmed007")

# # Properties for the product
# properties = {
#     "product_name": "Laptop",
#     "min_price": 999.99,
#     "max_price": 1999.99,
#     "description": "A high-end laptop with great performance.",
#     "categories": "Electronics, Computers",
#     "image": "/path/to/laptop_image.jpg",  # Image path
#     "main_description": ["Intel i7 processor", "16GB RAM", "512GB SSD"]
# }

# # Create the product node
# neo4j_conn.create_product_node("Product", properties)

# # neo4j_conn.delete_all_nodes()
# # Close the connection
# neo4j_conn.close()
