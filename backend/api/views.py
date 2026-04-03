import os
from django.conf import settings
from django.http import FileResponse, HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.tokens import RefreshToken

# cusotm view to serve video to dispkay properly for ios
def serve_media(request, path):
    file_path = os.path.join(settings.MEDIA_ROOT, path)
    if not os.path.exists(file_path):
        return HttpResponse(status=404)

    file_size = os.path.getsize(file_path)
    range_header = request.META.get("HTTP_RANGE")

    if range_header:
        range_val = range_header.strip().split("=")[1]
        start, end = range_val.split("-")
        start = int(start)
        end = int(end) if end else file_size - 1
        length = end - start + 1

        with open(file_path, "rb") as f:
            f.seek(start)
            data = f.read(length)

        response = HttpResponse(data, status=206, content_type="video/mp4")
        response["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        response["Accept-Ranges"] = "bytes"
        response["Content-Length"] = str(length)
        return response

    response = FileResponse(open(file_path, "rb"), content_type="video/mp4")
    response["Accept-Ranges"] = "bytes"
    response["Content-Length"] = str(file_size)
    return response

from .models import Analysis
from .serializers import AnalysisSerializer, RegisterSerializer, UserSerializer

SUPPORTED_EXERCISES = ["squat", "pushup", "deadlift"]


# return :  jwt + the new user object
class RegisterView(APIView):
    """
    POST /api/auth/register/
    Body: { username, email, password }
    """

    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        # pass to serialiser for validation
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        return Response(
            {
                "user": UserSerializer(user).data,
                "access": str(refresh.access_token),
                "refresh": str(refresh),
            },
            status=status.HTTP_201_CREATED,
        )


# return : current authenticated user profile
class MeView(APIView):
    """
    GET /api/auth/me/
    """

    def get(self, request):
        return Response(UserSerializer(request.user).data)


#  Save uploaded video + runs full pipeline + save back to DB +  return the complete analysis
class AnalyseView(APIView):
    """
    POST /api/analyse/
    Body: multipart/form-data with 'video' (file) and 'exercise_type' (string)
    """

    def post(self, request):
        video_file = request.FILES.get("video")
        exercise_type = request.data.get("exercise_type", "").lower().strip()

        if not video_file:
            return Response(
                {"error": "No video selected"}, status=status.HTTP_400_BAD_REQUEST
            )

        if exercise_type not in SUPPORTED_EXERCISES:
            return Response(
                {
                    "error": f"Invalid exercise type. Supported: {', '.join(SUPPORTED_EXERCISES)}"
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Save the analysis record -> store ID, status starts as 'processing'
        analysis = Analysis.objects.create(
            user=request.user,
            exercise_type=exercise_type,
            input_video=video_file,
        )

        try:
            # import here so Django can start without the model venv packages installed
            from exercise_video_analysis import analyse_video

            output_dir = str(settings.MEDIA_ROOT / "outputs")

            # CALL TO THE MAIN PIPELINE HERE!!!!!!!!!!!!!!!!!!!!!!!
            result = analyse_video(
                video_path=analysis.input_video.path,
                exercise_type=exercise_type,
                output_dir=output_dir,
                analysis_id=analysis.id,
            )

            if result is None:
                raise ValueError(
                    f"Exercise '{exercise_type}' is not fully implemented yet."
                )

            # store the output video path relative to MEDIA_ROOT to build url later
            output_abs = result["output_video_path"]
            output_rel = os.path.relpath(output_abs, settings.MEDIA_ROOT)

            # delete the raw input video
            raw_path = analysis.input_video.path
            analysis.input_video.delete(save=False)
            if os.path.exists(raw_path):
                os.remove(raw_path)

            analysis.output_video = output_rel
            analysis.rep_feedback = result["rep_feedback"]
            analysis.llm_summary = result["llm_summary"]
            analysis.total_reps = result["total_reps"]
            analysis.status = "complete"
            analysis.save()

        except Exception as e:
            analysis.status = "failed"
            analysis.error_message = str(e)
            analysis.save()
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        serializer = AnalysisSerializer(analysis, context={"request": request})
        return Response(serializer.data, status=status.HTTP_201_CREATED)


# returns a single analysis that belong to logged in user by id
class AnalysisDetailView(APIView):
    """
    GET /api/analysis/<id>/
    """

    def get(self, request, pk):
        try:
            analysis = Analysis.objects.get(pk=pk, user=request.user)
        except Analysis.DoesNotExist:
            return Response(
                {"error": "Analysis not found."}, status=status.HTTP_404_NOT_FOUND
            )

        serializer = AnalysisSerializer(analysis, context={"request": request})
        return Response(serializer.data)


# return all
class AnalysisListView(APIView):
    """
    GET /api/analyses/
    Returns all analyses for the current user, most recent first.
    """

    def get(self, request):
        analyses = Analysis.objects.filter(user=request.user).order_by("-created_at")
        serializer = AnalysisSerializer(
            analyses, many=True, context={"request": request}
        )
        return Response(serializer.data)
