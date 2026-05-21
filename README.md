# Keras Content Moderation System

Real deep learning portfolio project for text moderation. It trains a Keras neural network on the UCI SMS Spam Collection dataset, exports the learned vocabulary and weights, and runs actual model inference in the web app.

## Model

- Dataset: UCI SMS Spam Collection
- Framework: TensorFlow / Keras
- Architecture: bag-of-words input -> Dense(96, ReLU) -> Dropout -> Dense(32, ReLU) -> Dense(1, sigmoid)
- Test accuracy: 99.1%
- Test F1: 96.5%
- Exported artifacts:
  - `public/model/keras_sms_moderation.keras`
  - `public/model/moderation_model.json`
  - `reports/keras_moderation_metrics.json`

## Train

```bash
python scripts/train_keras_moderation.py
```

## Run

```bash
npm install
npm run dev
```

## Deploy

```bash
npm run build
vercel deploy --prod
```
