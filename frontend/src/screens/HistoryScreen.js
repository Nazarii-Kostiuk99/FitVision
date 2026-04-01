import React, { useState, useEffect } from "react";
import { View, Text, FlatList, TouchableOpacity } from "react-native";
import * as SecureStore from "expo-secure-store";
import BASE_URL from "../api/client";

export default function HistoryScreen({ navigation }) {
  const [analyses, setAnalyses] = useState([]);

  useEffect(() => {
    loadHistory();
  }, []);

  async function loadHistory() {
    try {
      const token = await SecureStore.getItemAsync("access_token");
      const response = await fetch(`${BASE_URL}/analyses/`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await response.json();
      setAnalyses(data);
    } catch (error) {
      console.log(error);
    }
  }

  return (
    <View className="flex-1 bg-black px-6 pt-10">
      <Text className="text-turquoise text-2xl font-bold mb-1">History</Text>
      <Text className="text-zinc-400 mb-6">Your past sessions</Text>

      <FlatList
        data={analyses}
        keyExtractor={(item) => item.id.toString()}
        ListEmptyComponent={
          <Text className="text-zinc-500 text-center mt-10">No analyses yet</Text>
        }
        renderItem={({ item }) => (
          <TouchableOpacity
            onPress={() => navigation.navigate("Results", { analysis: item })}
            className="bg-zinc-900 rounded-lg p-4 mb-3 border border-zinc-800"
          >
            <View className="flex-row justify-between items-center mb-1">
              <Text className="text-white font-semibold capitalize">{item.exercise_type}</Text>
              <Text className="text-turquoise font-bold">{item.total_reps} reps</Text>
            </View>
            <Text className="text-zinc-500 text-xs">{item.created_at}</Text>
          </TouchableOpacity>
        )}
      />
    </View>
  );
}
