import React, { useState, useEffect } from "react";
import { View, Text, FlatList, TouchableOpacity, Alert } from "react-native";
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

  async function deleteAccount() {
    Alert.alert(
      "Delete account",
      "This will permanently delete your account and all associated records. This cannot be undone.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: async () => {
            try {
              const token = await SecureStore.getItemAsync("access_token");
              await fetch(`${BASE_URL}/auth/account/`, {
                method: "DELETE",
                headers: { Authorization: `Bearer ${token}` },
              });
              await SecureStore.deleteItemAsync("access_token");
              navigation.replace("Login");
            } catch (error) {
              Alert.alert("Error", error.message);
            }
          },
        },
      ]
    );
  }

  async function deleteAnalysis(id) {
    Alert.alert("Delete record", "Are you sure you want to delete this record?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: async () => {
          try {
            const token = await SecureStore.getItemAsync("access_token");
            await fetch(`${BASE_URL}/analysis/${id}/`, {
              method: "DELETE",
              headers: { Authorization: `Bearer ${token}` },
            });
            setAnalyses((prev) => prev.filter((a) => a.id !== id));
          } catch (error) {
            Alert.alert("Error", error.message);
          }
        },
      },
    ]);
  }

  return (
    <View className="flex-1 bg-black px-6 pt-10">
      <Text className="text-turquoise text-2xl font-bold mb-1">History</Text>
      <Text className="text-zinc-400 mb-6">Your past sessions</Text>

      <FlatList
        style={{ flex: 1 }}
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
            <View className="flex-row justify-between items-center">
              <Text className="text-zinc-500 text-xs">{item.created_at}</Text>
              <TouchableOpacity onPress={() => deleteAnalysis(item.id)}>
                <Text className="text-red-500 text-xs">Delete</Text>
              </TouchableOpacity>
            </View>
          </TouchableOpacity>
        )}
      />

      <TouchableOpacity onPress={deleteAccount} className="py-4 items-center">
        <Text className="text-red-500 text-sm">Delete account and all data</Text>
      </TouchableOpacity>
    </View>
  );
}
