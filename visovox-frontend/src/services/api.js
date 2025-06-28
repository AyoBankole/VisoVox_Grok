const BASE_URL = import.meta.env.VITE_BACKEND_URL;

export const uploadImage = async (formData, type) => {
  const endpoint = `${BASE_URL}/${type}`;
  const response = await fetch(endpoint, {
    method: 'POST',
    body: formData,
  });
  const data = await response.json();
  return data;
};