import React, { useState } from "react";
import { View, Text, TextInput, TouchableOpacity, Alert } from "react-native";

export default function RegisterScreen({ navigation }) {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  function handleRegister() {
    if (!username || !email || !password) {
      Alert.alert("Please fill in all fields");
      return;
    }
    navigation.navigate("Terms", { username, email, password });
  }

  return (
    <View className="flex-1 bg-black px-6 justify-center">
      <Text className="text-turquoise text-4xl font-bold mb-2">FitVision</Text>
      <Text className="text-white text-lg mb-8">Create an account</Text>

      <Text className="text-white mb-1">Username</Text>
      <TextInput
        value={username}
        onChangeText={setUsername}
        autoCapitalize="none"
        placeholderTextColor="#555"
        placeholder="Enter username"
        className="bg-zinc-900 text-white border border-zinc-700 rounded-lg px-4 py-3 mb-4"
      />

      <Text className="text-white mb-1">Email</Text>
      <TextInput
        value={email}
        onChangeText={setEmail}
        autoCapitalize="none"
        placeholderTextColor="#555"
        placeholder="Enter email"
        className="bg-zinc-900 text-white border border-zinc-700 rounded-lg px-4 py-3 mb-4"
      />

      <Text className="text-white mb-1">Password</Text>
      <TextInput
        value={password}
        onChangeText={setPassword}
        secureTextEntry
        placeholderTextColor="#555"
        placeholder="Min 8 characters"
        className="bg-zinc-900 text-white border border-zinc-700 rounded-lg px-4 py-3 mb-6"
      />

      <TouchableOpacity
        onPress={handleRegister}
        className="bg-turquoise rounded-lg py-3 items-center mb-3"
      >
        <Text className="text-black font-bold text-base">Register</Text>
      </TouchableOpacity>

      <TouchableOpacity onPress={() => navigation.navigate("Login")}>
        <Text className="text-zinc-400 text-center">
          Already have an account?{" "}
          <Text className="text-turquoise">Login</Text>
        </Text>
      </TouchableOpacity>
    </View>
  );
}