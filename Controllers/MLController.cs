using Microsoft.AspNetCore.Mvc;
using MottuApi.Services.Interfaces;
using Microsoft.AspNetCore.Authorization; 

namespace MottuApi.Controllers
{
    /// <summary>
    /// Controller para preços de diárias.
    /// </summary>
    [Authorize]
    [ApiController]
    [Route("api/v{version:apiVersion}/[controller]")]
    [ApiVersion("1.0")] 
    public class MLController : ControllerBase
    {
        private readonly IMLPredictionService _mlService;

        public MLController(IMLPredictionService mlService)
        {
            _mlService = mlService;
        }

        /// <summary>
        /// Preve o valor de diária de locação para uma nova moto usando um modelo de Machine Learning.
        /// </summary>
        /// <param name="ano">Ano de fabricação da moto.</param>
        /// <param name="quilometragem">Quilometragem atual da moto.</param>
        /// <returns>Um objeto com a previsão.</returns>
        [HttpGet("prever-diaria")]
        public ActionResult PredictDiaria(float ano, float quilometragem)
        {
            if (ano < 2017 || quilometragem < 0)
            {
                return BadRequest("O modelo só é treinado para motos a partir do ano 2017 e KM não pode ser negativa.");
            }

            try
            {
                var predictedValue = _mlService.PredictValorDiaria(ano, quilometragem);
                
                return Ok(new 
                {
                    InputAno = ano,
                    InputQuilometragem = quilometragem,
                    PrevisaoDiaria = predictedValue,
                    ModeloUsado = "Regressão ML.NET (FastTreeTweedie)" 
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Erro interno ao realizar previsão: {ex.Message}");
            }
        }
    }
}