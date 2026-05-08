from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
import os
from django.conf import settings
from api.models import Analysis


# delete analysis + vids > 30 days"
class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        cutoff = timezone.now() - timedelta(days=30)
        old = Analysis.objects.filter(created_at__lt=cutoff)
        for a in old:
            if a.output_video:
                path = os.path.join(settings.MEDIA_ROOT, str(a.output_video))
                if os.path.exists(path):
                    os.remove(path)
                a.output_video = None
                a.save()
        self.stdout.write(
            f"Cleared videos for {old.count()} analyses older than 30 days."
        )
