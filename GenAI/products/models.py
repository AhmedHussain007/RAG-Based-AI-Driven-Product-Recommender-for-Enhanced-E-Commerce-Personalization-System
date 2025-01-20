from django.db import models
from django.contrib.auth.models import User

class Product(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    min_price = models.DecimalField(max_digits=10, decimal_places=2)
    max_price = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField()
    categories = models.CharField(max_length=255)
    image = models.ImageField(upload_to='product_images/')
    tags_list = models.CharField(max_length=255,default=None)


    def save(self, *args, **kwargs):
        # First, save the product to the default database (SQL)
        super().save(*args, **kwargs)

        min_price = float(self.min_price)
        max_price = float(self.max_price)
        # After saving to SQL, create the product in Neo4j
        from products.Neo4jDB import Neo4jManager
        neo4j_manager = Neo4jManager(uri='bolt://localhost:7687', user='neo4j', password='Ahmed007')
        neo4j_manager.create_product_node(
            product_id = self.id,
            name=self.name,
            min_price=min_price,
            max_price=max_price,
            description=self.description,
            categories=self.categories,
            image=self.image.url,  # Assuming you want to store the image URL
            tags_list=self.tags_list
        )

    def __str__(self):
        return str(self.id) + " " + self.name

class UserProductClick(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey('Product', on_delete=models.CASCADE)
    clicked_at = models.DateTimeField(auto_now_add=True)