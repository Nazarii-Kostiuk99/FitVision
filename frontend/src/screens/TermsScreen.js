import React from "react";
import { View, Text, ScrollView, TouchableOpacity, Alert } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import * as SecureStore from "expo-secure-store";
import BASE_URL from "../api/client";

export default function TermsScreen({ navigation, route }) {
  const { username, email, password } = route.params;
  const insets = useSafeAreaInsets();

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
    <View
      className="flex-1 bg-black px-6"
      style={{ paddingTop: insets.top + 16, paddingBottom: insets.bottom + 16 }}
    >
      <Text className="text-turquoise text-2xl font-bold mb-1">
        Health & Privacy Notice
      </Text>
      <Text className="text-zinc-400 text-xs mb-6">
        Please read and accept before creating your account.
      </Text>

      <ScrollView
        className="flex-1 mb-6"
        showsVerticalScrollIndicator={false}
        contentContainerStyle={{ paddingBottom: 16 }}
      >
        <Text className="text-zinc-300 text-sm leading-7">
          FitVision provides general guidance only and is not a replacement for
          professional coaching or medical advice. If you have a pre-existing
          injury or health condition, consult a qualified professional before
          use. Results may vary slightly depending on camera placement and body
          proportions.
        </Text>

        <Text className="text-turquoise text-xs font-semibold mt-6 mb-2 uppercase tracking-widest">
          Your Data
        </Text>
        <Text className="text-zinc-300 text-sm leading-7">
          All uploaded video footage is anonymised immediately upon receipt and
          raw footage is permanently deleted. Anonymised videos and analysis
          records are retained for up to 30 days, after which they are
          automatically removed. You may delete your account and all associated
          records at any time.
        </Text>

        <Text className="text-turquoise text-xs font-semibold mt-6 mb-2 uppercase tracking-widest">
          Disclaimer
        </Text>
        <Text className="text-zinc-300 text-sm leading-7">
          Use this application at your own discretion. FitVision is not liable
          for any injury or harm resulting from the use of this application.
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
        className="py-3 items-center"
      >
        <Text className="text-zinc-400">Decline</Text>
      </TouchableOpacity>
    </View>
  );
}
