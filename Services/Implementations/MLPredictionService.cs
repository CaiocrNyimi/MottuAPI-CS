using Microsoft.ML;
using MottuApi.Services.Interfaces; 
using MottuApi.MLModels;
using System;
using System.IO;

namespace MottuApi.Services.Implementations
{
    public class MLPredictionService : IMLPredictionService
    {
        private readonly PredictionEngine<MotoData, MotoPrediction> _predictionEngine;

        public MLPredictionService()
        {
            var mlContext = new MLContext();
            
            const string ModelPath = "MotoModel.zip";

            if (!File.Exists(ModelPath))
            {
                throw new FileNotFoundException(
                    $"O arquivo do modelo ML '{ModelPath}' não foi encontrado. Execute o ModelTrainer uma vez."
                );
            }

            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);
            
            _predictionEngine = mlContext.Model.CreatePredictionEngine<MotoData, MotoPrediction>(trainedModel);
        }

        public float PredictValorDiaria(float ano, float quilometragem)
        {
            var idade = (float)DateTime.Now.Year - ano;
            
            var dataSample = new MotoData 
            { 
                Ano = ano, 
                Quilometragem = quilometragem,
                Idade = idade
            };

            var prediction = _predictionEngine.Predict(dataSample);

            return (float)Math.Round(prediction.PredictedValorDiaria, 2);
        }
    }
}