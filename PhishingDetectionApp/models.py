from django.db import models

class BlockedUrl(models.Model):
    url = models.CharField(max_length=500)
    blocked_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.url
