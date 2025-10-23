using Microsoft.ML.Data;

namespace MottuApi.MLModels
{
    public class MotoData
    {
        [LoadColumn(0)]
        public float Ano { get; set; }

        [LoadColumn(1)]
        public float Quilometragem { get; set; }

        public float Idade { get; set; }

        [LoadColumn(2), ColumnName("Label")]
        public float ValorDiaria { get; set; }
    }

    public class MotoPrediction
    {
        [ColumnName("Score")]
        public float PredictedValorDiaria { get; set; }
    }
}