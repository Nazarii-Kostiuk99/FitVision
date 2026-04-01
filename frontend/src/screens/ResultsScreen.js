import React from "react";
import { View, Text, ScrollView, TouchableOpacity } from "react-native";
import { VideoView, useVideoPlayer } from "expo-video";

export default function ResultsScreen({ route, navigation }) {
  const { analysis } = route.params;
  console.log("video url:", analysis.output_video_url);

  const player = useVideoPlayer(analysis.output_video_url, (p) => {
    p.loop = true;
    p.play();
  });

  return (
    <ScrollView className="flex-1 bg-black px-6 pt-6">
      <Text className="text-turquoise text-2xl font-bold mb-1">Results</Text>
      <Text className="text-zinc-400 mb-6 capitalize">{analysis.exercise_type}</Text>

      <View className="flex-row gap-3 mb-6">
        <View className="flex-1 bg-zinc-900 rounded-lg p-4 items-center">
          <Text className="text-turquoise text-2xl font-bold">{analysis.total_reps}</Text>
          <Text className="text-zinc-400 text-xs mt-1">Reps</Text>
        </View>
        <View className="flex-1 bg-zinc-900 rounded-lg p-4 items-center">
          <Text className="text-turquoise text-2xl font-bold capitalize">{analysis.status}</Text>
          <Text className="text-zinc-400 text-xs mt-1">Status</Text>
        </View>
      </View>

      {analysis.output_video_url && (
        <VideoView
          player={player}
          style={{ width: "100%", height: 220, borderRadius: 10, marginBottom: 24 }}
          nativeControls={true}
          allowsFullscreen={true}
        />
      )}

      <Text className="text-white font-semibold mb-3">Rep Feedback</Text>
      {analysis.rep_feedback.map((block, i) => (
        <View key={i} className="bg-zinc-900 rounded-lg p-4 mb-3">
          <Text className="text-turquoise font-semibold mb-2">
            Rep {block.rep} · {block.side}
          </Text>
          {block.feedback.map((line, j) => (
            <Text key={j} className="text-zinc-300 text-sm mb-1">• {line}</Text>
          ))}
        </View>
      ))}

      <Text className="text-white font-semibold mt-2 mb-3">AI Summary</Text>
      <View className="bg-zinc-900 rounded-lg p-4 mb-8">
        <Text className="text-zinc-300 text-sm leading-5">{analysis.llm_summary}</Text>
      </View>

      <TouchableOpacity
        onPress={() => navigation.navigate("Home")}
        className="bg-turquoise rounded-lg py-4 items-center mb-10"
      >
        <Text className="text-black font-bold">Back to Home</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}
