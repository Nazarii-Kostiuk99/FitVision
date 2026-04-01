import React, { useState } from "react";
import { View, Text, TouchableOpacity, Alert, ActivityIndicator } from "react-native";
import * as ImagePicker from "expo-image-picker";
import * as SecureStore from "expo-secure-store";
import BASE_URL from "../api/client";

const EXERCISES = ["squat", "pushup", "deadlift"];

export default function HomeScreen({ navigation }) {
  const [videoUri, setVideoUri] = useState(null);
  const [exercise, setExercise] = useState("squat");
  const [loading, setLoading] = useState(false);

  async function pickVideo() {
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

    setLoading(true);

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

      if (data.id) {
        navigation.navigate("Results", { analysis: data });
      } else {
        Alert.alert("Something went wrong", JSON.stringify(data));
      }
    } catch (error) {
      Alert.alert("Error", error.message);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <View className="flex-1 bg-black items-center justify-center px-6">
        <ActivityIndicator size="large" color="#40E0D0" />
        <Text className="text-white text-center mt-4">Processing video...</Text>
        <Text className="text-zinc-500 text-center mt-1">This may take a few minutes</Text>
      </View>
    );
  }

  return (
    <View className="flex-1 bg-black px-6 pt-10">
      <Text className="text-turquoise text-3xl font-bold mb-1">FitVision</Text>
      <Text className="text-zinc-400 mb-8">Analyse your form</Text>

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
            <Text className={exercise === ex ? "text-black font-bold" : "text-white"}>
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

      <TouchableOpacity onPress={() => navigation.navigate("History")} className="py-3 items-center">
        <Text className="text-zinc-400">View History</Text>
      </TouchableOpacity>
    </View>
  );
}
