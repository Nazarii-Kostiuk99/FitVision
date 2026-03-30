from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Analysis


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = ["id", "username", "email", "password"]

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "email"]


class AnalysisSerializer(serializers.ModelSerializer):
    output_video_url = (
        serializers.SerializerMethodField()
    )  # calls get_outpuy_vieo_url to compute since not a model field

    class Meta:
        model = Analysis
        fields = [
            "id",
            "exercise_type",
            "status",
            "total_reps",
            "rep_feedback",
            "llm_summary",
            "output_video_url",
            "created_at",
            "error_message",
        ]

    def get_output_video_url(self, obj):
        if not obj.output_video:
            return None
        request = self.context.get("request")
        return request.build_absolute_uri(obj.output_video.url)
