import React from "react";
import { View, Text, ScrollView, TouchableOpacity, Alert } from "react-native";
import * as SecureStore from "expo-secure-store";
import BASE_URL from "../api/client";

export default function TermsScreen({ navigation, route }) {
  const { username, email, password } = route.params;

  async function handleAccept() {
    try {
      const response = await fetch(`${BASE_URL}/auth/register/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password }),
      });

      const data = await response.json();

      if (data.access) {
        await SecureStore.setItemAsync("access_token", data.access);
        navigation.replace("Home");
      } else {
        Alert.alert("Register failed", JSON.stringify(data));
      }
    } catch (error) {
      Alert.alert("Error", error.message);
    }
  }

  function handleDecline() {
    navigation.goBack();
  }

  return (
    <View className="flex-1 bg-black px-6 pt-12">
      <Text className="text-turquoise text-2xl font-bold mb-1">
        Terms & Conditions
      </Text>
      <Text className="text-zinc-400 text-xs mb-4">
        Please read and accept before creating your account.
      </Text>

      <ScrollView className="flex-1 mb-6">
        <Text className="text-white text-sm leading-6">
          TODO: ADD TERMS + PRIVACY STUFF
        </Text>
      </ScrollView>

      <TouchableOpacity
        onPress={handleAccept}
        className="bg-turquoise rounded-lg py-3 items-center mb-3"
      >
        <Text className="text-black font-bold text-base">I Accept</Text>
      </TouchableOpacity>

      <TouchableOpacity
        onPress={handleDecline}
        className="py-3 items-center mb-6"
      >
        <Text className="text-zinc-400">Decline</Text>
      </TouchableOpacity>
    </View>
  );
}
