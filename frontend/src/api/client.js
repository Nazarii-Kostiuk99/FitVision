import * as SecureStore from "expo-secure-store";

// change this to your machine's IP when testing on a real device
const BASE_URL = "http://10.136.8.6:8000/api";

export default BASE_URL;

// to get token from storag
export async function getToken() {
  const token = await SecureStore.getItemAsync("access_token");
  return token;
}
