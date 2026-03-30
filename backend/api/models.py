from django.db import models
from django.contrib.auth.models import User


class Analysis(models.Model):
    EXERCISE_CHOICES = [
        ("squat", "Squat"),
        ("pushup", "Pushup"),
        ("deadlift", "Deadlift"),
    ]
    STATUS_CHOICES = [
        ("processing", "Processing"),
        ("complete", "Complete"),
        ("failed", "Failed"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="analyses")
    exercise_type = models.CharField(max_length=50, choices=EXERCISE_CHOICES)

    # to be deldeted later after processing, not stored in db
    input_video = models.FileField(upload_to="uploads/", null=True, blank=True)

    # final analysis video, null untill finsihed processing
    output_video = models.FileField(upload_to="outputs/", null=True, blank=True)

    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="processing"
    )

    # per rep feedback main pipeline (json)
    rep_feedback = models.JSONField(default=list)

    # qualitative llama feedback summary
    llm_summary = models.TextField(blank=True)

    total_reps = models.IntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    error_message = models.TextField(blank=True)

    def __str__(self):
        return f"{self.user.username} — {self.exercise_type} — {self.status} ({self.created_at:%Y-%m-%d %H:%M})"
