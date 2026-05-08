import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  Alert,
  Animated,
  useWindowDimensions,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import * as ImagePicker from "expo-image-picker";
import * as SecureStore from "expo-secure-store";
import BASE_URL from "../api/client";

const EXERCISES = ["Squat", "Pushup", "Deadlift"];

// simulates tqdm but without websockets, quick to 30, slow 30-90, quick 90-100
function ProgressBar({ progress }) {
  const { width: screenWidth } = useWindowDimensions();
  const barWidth = screenWidth - 48;
  const animatedWidth = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(animatedWidth, {
      toValue: (progress / 100) * barWidth,
      duration: progress === 100 ? 300 : 800,
      useNativeDriver: false,
    }).start();
  }, [progress]);

  return (
    <View style={{ width: barWidth }}>
      <View
        style={{
          width: barWidth,
          height: 6,
          backgroundColor: "#27272a",
          borderRadius: 3,
          overflow: "hidden",
        }}
      >
        <Animated.View
          style={{
            width: animatedWidth,
            height: 6,
            backgroundColor: "#40E0D0",
            borderRadius: 3,
          }}
        />
      </View>
      <Text
        style={{
          color: "#40E0D0",
          marginTop: 8,
          textAlign: "right",
          fontSize: 13,
        }}
      >
        {Math.round(progress)}%
      </Text>
    </View>
  );
}

export default function HomeScreen({ navigation }) {
  const insets = useSafeAreaInsets();
  const [videoUri, setVideoUri] = useState(null);
  const [exercise, setExercise] = useState("squat");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState("Uploading...");
  const crawlAnim = useRef(null);

  async function pickVideo() {
    // WITHOUT THIS PERMISSION -> INSTA CRASH
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert(
        "Permission required",
        "Please allow access to your photo library.",
      );
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["videos"],
    });

    if (!result.canceled) {
      setVideoUri(result.assets[0].uri);
    }
  }

  async function handleSubmit() {
    if (!videoUri) {
      Alert.alert("Please select a video first");
      return;
    }

    setProgress(0);
    setStatusText("Uploading...");
    setLoading(true);

    // jump to 30 quickly aka upload
    setTimeout(() => setProgress(30), 400);
    // slowly  30 -> 90 over 90s (processing)
    setTimeout(() => {
      setStatusText("Analysing your form...");
      let current = 30;
      crawlAnim.current = setInterval(() => {
        current += 0.7;
        if (current >= 90) {
          clearInterval(crawlAnim.current);
          current = 90;
        }
        setProgress(current);
      }, 700);
    }, 1200);

    try {
      const token = await SecureStore.getItemAsync("access_token");

      const form = new FormData();
      form.append("exercise_type", exercise);
      form.append("video", {
        uri: videoUri,
        name: "video.mp4",
        type: "video/mp4",
      });

      const response = await fetch(`${BASE_URL}/analyse/`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      });

      const data = await response.json();

      //  snap to 100%
      clearInterval(crawlAnim.current);
      setStatusText("Done!");
      setProgress(100);

      setTimeout(() => {
        setLoading(false);
        if (data.id) {
          navigation.navigate("Results", { analysis: data });
        } else {
          Alert.alert("Something went wrong", JSON.stringify(data));
        }
      }, 600);
    } catch (error) {
      clearInterval(crawlAnim.current);
      setLoading(false);
      Alert.alert("Error", error.message);
    }
  }

  if (loading) {
    return (
      <View className="flex-1 bg-black items-center justify-center px-6">
        <Text className="text-turquoise text-xl font-bold mb-2">FitVision</Text>
        <Text className="text-white text-center mb-6">{statusText}</Text>
        <ProgressBar progress={progress} />
        <Text className="text-zinc-500 text-center text-sm mb-6">
          This may take a few minutes
        </Text>
      </View>
    );
  }

  return (
    <View
      className="flex-1 bg-black px-6"
      style={{ paddingTop: insets.top, paddingBottom: insets.bottom + 16 }}
    >
      <View className="flex-1 justify-center">
        <Text className="text-turquoise text-6xl font-bold text-center">
          FitVision
        </Text>
        <Text className="text-zinc-400 mt-1 text-xl text-center mb-12">
          Analyse your form
        </Text>

        <Text className="text-white font-semibold mb-3">Select Exercise</Text>
        <View className="flex-row gap-3 mb-8">
          {EXERCISES.map((ex) => (
            <TouchableOpacity
              key={ex}
              onPress={() => setExercise(ex)}
              className={`flex-1 py-3 rounded-lg items-center border ${
                exercise === ex
                  ? "bg-turquoise border-turquoise"
                  : "bg-zinc-900 border-zinc-700"
              }`}
            >
              <Text
                className={
                  exercise === ex ? "text-black font-bold" : "text-white"
                }
              >
                {ex}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <TouchableOpacity
          onPress={pickVideo}
          className="bg-zinc-900 border border-zinc-700 rounded-lg py-4 items-center mb-3"
        >
          <Text className="text-turquoise font-semibold">
            {videoUri ? "Change Video" : "Pick Video"}
          </Text>
          {videoUri && (
            <Text className="text-zinc-500 text-xs mt-1">Video selected</Text>
          )}
        </TouchableOpacity>

        <TouchableOpacity
          onPress={handleSubmit}
          className="bg-turquoise rounded-lg py-4 items-center mb-3"
        >
          <Text className="text-black font-bold text-base">Analyse</Text>
        </TouchableOpacity>

        <TouchableOpacity
          onPress={() => navigation.navigate("History")}
          className="py-3 items-center"
        >
          <Text className="text-zinc-400">View History</Text>
        </TouchableOpacity>
      </View>

      <View className="bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-3 mb-3 flex-row items-center">
        <Text className="text-yellow-400 mr-2 text-sm">⚠</Text>
        <Text className="text-zinc-400 text-xs flex-1">
          Position yourself approximately 90° to the camera at a distance of
          1.5–2 m, ensuring your full body is in frame. For best results, record
          in a clear space with as few people around as possible.
        </Text>
      </View>
      <View className="bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-3 flex-row items-center">
        <Text className="text-zinc-500 mr-2 text-sm">ℹ</Text>
        <Text className="text-zinc-500 text-xs flex-1">
          FitVision provides general guidance only and is not a replacement for
          professional coaching advice. If you have a pre-existing injury or
          health condition, consult a qualified professional before use.
        </Text>
      </View>
    </View>
  );
}
