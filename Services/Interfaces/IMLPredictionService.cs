namespace MottuApi.Services.Interfaces
{
    public interface IMLPredictionService
    {
        float PredictValorDiaria(float ano, float quilometragem);
    }
}