import React, { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, Alert } from "react-native";
import * as SecureStore from "expo-secure-store";
import BASE_URL from "../api/client";

export default function LoginScreen({ navigation }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  async function handleLogin() {
    try {
      const response = await fetch(`${BASE_URL}/auth/login/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json();

      if (data.access) {
        await SecureStore.setItemAsync("access_token", data.access);
        navigation.replace("Home");
      } else {
        Alert.alert("Login failed", JSON.stringify(data));
      }
    } catch (error) {
      Alert.alert("Error", error.message);
    }
  }

  return (
    <View className="flex-1 bg-black px-6 justify-center">
      <Text className="text-turquoise text-4xl font-bold mb-2">FitVision</Text>
      <Text className="text-white text-lg mb-8">Sign in to continue</Text>

      <Text className="text-white mb-1">Username</Text>
      <TextInput
        value={username}
        onChangeText={setUsername}
        autoCapitalize="none"
        placeholderTextColor="#555"
        placeholder="Enter username"
        className="bg-zinc-900 text-white border border-zinc-700 rounded-lg px-4 py-3 mb-4"
      />

      <Text className="text-white mb-1">Password</Text>
      <TextInput
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        placeholderTextColor="#555"
        placeholder="Enter password"
        className="bg-zinc-900 text-white border border-zinc-700 rounded-lg px-4 py-3 mb-6"
      />

      <TouchableOpacity
        onPress={handleLogin}
        className="bg-turquoise rounded-lg py-3 items-center mb-3"
      >
        <Text className="text-black font-bold text-base">Login</Text>
      </TouchableOpacity>

      <TouchableOpacity onPress={() => navigation.navigate("Register")}>
        <Text className="text-zinc-400 text-center">
          No account?{" "}
          <Text className="text-turquoise">Register</Text>
        </Text>
      </TouchableOpacity>
    </View>
  );
}
