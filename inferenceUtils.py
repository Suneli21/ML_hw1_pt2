import os
import pickle


model_path = os.path.join(r'models/linear_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✅ Модель сохранена: {model_path}")
