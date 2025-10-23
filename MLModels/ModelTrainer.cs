using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Collections.Generic;
using System.Linq; 
using MottuApi.MLModels;
using System.IO;
using System;

public static class ModelTrainer
{
    public const string ModelPath = "MotoModel.zip";

    public static void TrainAndSaveModelIfNeeded()
    {
        if (File.Exists(ModelPath))
        {
            Console.WriteLine($"\nModelo ML.NET já existe em: {Path.GetFullPath(ModelPath)}");
            Console.WriteLine("--------------------------------------------------------------------------------\n");
            return;
        }

        Console.WriteLine("Iniciando treinamento do modelo ML.NET...");

        var mlContext = new MLContext();

        var sampleData = new List<MotoData>
        {
            new MotoData { Ano = 2024, Quilometragem = 100, ValorDiaria = 150.00f },
            new MotoData { Ano = 2024, Quilometragem = 500, ValorDiaria = 148.00f },
            new MotoData { Ano = 2023, Quilometragem = 2000, ValorDiaria = 135.00f },
            new MotoData { Ano = 2023, Quilometragem = 5000, ValorDiaria = 125.00f },
            new MotoData { Ano = 2023, Quilometragem = 8000, ValorDiaria = 120.00f },
            new MotoData { Ano = 2022, Quilometragem = 10000, ValorDiaria = 115.00f },
            new MotoData { Ano = 2022, Quilometragem = 15000, ValorDiaria = 105.50f },
            new MotoData { Ano = 2022, Quilometragem = 20000, ValorDiaria = 98.00f },
            new MotoData { Ano = 2021, Quilometragem = 25000, ValorDiaria = 95.00f },
            new MotoData { Ano = 2021, Quilometragem = 30000, ValorDiaria = 90.00f },
            new MotoData { Ano = 2021, Quilometragem = 35000, ValorDiaria = 85.00f },
            new MotoData { Ano = 2020, Quilometragem = 40000, ValorDiaria = 80.00f },
            new MotoData { Ano = 2020, Quilometragem = 50000, ValorDiaria = 75.00f },
            new MotoData { Ano = 2020, Quilometragem = 60000, ValorDiaria = 70.00f },
            new MotoData { Ano = 2019, Quilometragem = 65000, ValorDiaria = 65.00f },
            new MotoData { Ano = 2019, Quilometragem = 70000, ValorDiaria = 60.00f },
            new MotoData { Ano = 2019, Quilometragem = 75000, ValorDiaria = 58.00f },
            new MotoData { Ano = 2018, Quilometragem = 80000, ValorDiaria = 55.00f },
            new MotoData { Ano = 2018, Quilometragem = 85000, ValorDiaria = 52.00f },
            new MotoData { Ano = 2018, Quilometragem = 90000, ValorDiaria = 48.00f },
            new MotoData { Ano = 2017, Quilometragem = 95000, ValorDiaria = 46.00f },
            new MotoData { Ano = 2017, Quilometragem = 100000, ValorDiaria = 45.00f },
            new MotoData { Ano = 2016, Quilometragem = 110000, ValorDiaria = 40.00f },
            new MotoData { Ano = 2016, Quilometragem = 120000, ValorDiaria = 35.00f },
        };

        var trainingDataWithFeatures = sampleData.Select(data => new MotoData 
        {
            Ano = data.Ano,
            Quilometragem = data.Quilometragem,
            ValorDiaria = data.ValorDiaria,
            Idade = (float)DateTime.Now.Year - data.Ano 
        }).ToList();

        IDataView trainingDataView = mlContext.Data.LoadFromEnumerable(trainingDataWithFeatures);

        var dataProcessPipeline = mlContext.Transforms.NormalizeMinMax(
            outputColumnName: nameof(MotoData.Idade),
            inputColumnName: nameof(MotoData.Idade)
        )
        .Append(mlContext.Transforms.NormalizeLogMeanVariance(
            outputColumnName: nameof(MotoData.Quilometragem), 
            inputColumnName: nameof(MotoData.Quilometragem)
        ))
        .Append(mlContext.Transforms.Concatenate("Features",
            nameof(MotoData.Idade),
            nameof(MotoData.Quilometragem)
        ));

        var trainer = mlContext.Regression.Trainers.LightGbm(labelColumnName: "Label", featureColumnName: "Features", numberOfIterations: 200);
        var trainingPipeline = dataProcessPipeline.Append(trainer);
        var trainedModel = trainingPipeline.Fit(trainingDataView);

        mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

        Console.WriteLine($"\nModelo ML.NET treinado e salvo em: {Path.GetFullPath(ModelPath)}");
        Console.WriteLine("--------------------------------------------------------------------------------\n");
    }
}