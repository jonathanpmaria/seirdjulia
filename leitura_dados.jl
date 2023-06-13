using DataFrames, CSV, Dates, PyPlot

function le_SIVEP_Gripe(nome_arquivo_csv, id) 
    df = DataFrame(CSV.File(nome_arquivo_csv; select=["DT_SIN_PRI","CO_MUN_NOT","CLASSI_FIN","EVOLUCAO","DT_EVOLUCA"], dateformat="dd/mm/yyyy"))
    df = dropmissing(copy(df), "DT_EVOLUCA")
    if (id == 0)
        dinit = Date(2020,2,26)
        dend = Date(2020,12,31)
    elseif (id == 1)
        dinit = Date(2021,1,1)
        dend = Date(2021,12,31)
    elseif (id == 2)
        dinit = Date(2022,1,1)
        dend = Date(2022,5,30)
    end
    n_days = Dates.value(dend - dinit) + 1
    D_obs = zeros(Int,n_days)

    for i = 1:size(df,1)
        if (df[i,2] === 355030) #código notif. SP capital
            if (df[i,3] === 4 || df[i,3] === 5) #SRAG de causa desconhecida (4) ou por COVID-19 (5)
                if (df[i,4] === 2) #óbito
                    if (df[i,1] >= dinit && df[i,5] <= dend)
                        ind = Dates.value(df[i,5] - dinit) + 1
                        D_obs[ind] = D_obs[ind] + 1
                    end
                end
            end
        end
    end

    println("Mortes: ", sum(D_obs))
    println("Número de dias: ", n_days)
    return D_obs
end

function le_SIVEP_Gripe_2(csv1, csv2, id) 
    df21 = DataFrame(CSV.File(csv1; select=["DT_SIN_PRI","CO_MUN_NOT","CLASSI_FIN","EVOLUCAO","DT_EVOLUCA"], dateformat="dd/mm/yyyy"))
    df22 = DataFrame(CSV.File(csv2; select=["DT_SIN_PRI","CO_MUN_NOT","CLASSI_FIN","EVOLUCAO","DT_EVOLUCA"], dateformat="dd/mm/yyyy"))
    df21 = dropmissing(copy(df21), "DT_EVOLUCA")
    df22 = dropmissing(copy(df22), "DT_EVOLUCA")
    if (id == 1)
        dinit1 = Date(2020,11,11)
        dinit2 = Date(2021,1,1)
        dend = Date(2021,12,31)
    elseif (id == 2)
        dinit1 = Date(2021,11,25)
        dinit2 = Date(2022,1,1)
        #dend = Date(2022,3,4) #100 dias
        dend = Date(2022,5,23) #180 dias
        #dend = Date(2022,5,30) #187 dias
    end
    n_days = Dates.value(dend - dinit1) + 1
    D_obs = zeros(Int,n_days)

    for i = 1:size(df21,1)
        if (df21[i,2] === 355030) #código notif. SP capital
            if (df21[i,3] === 4 || df21[i,3] === 5) #SRAG de causa desconhecida (4) ou por COVID-19 (5)
                if (df21[i,4] === 2) #óbito
                    if (df21[i,1] >= dinit1 && df21[i,5] <= dend)
                        ind = Dates.value(df21[i,5] - dinit1) + 1
                        D_obs[ind] = D_obs[ind] + 1
                    end
                end
            end
        end
    end

    for i = 1:size(df22,1)
        if (df22[i,2] === 355030) #código notif. SP capital
            if (df22[i,3] === 4 || df22[i,3] === 5) #SRAG de causa desconhecida (4) ou por COVID-19 (5)
                if (df22[i,4] === 2) #óbito
                    if (df22[i,1] >= dinit2 && df22[i,5] <= dend)
                        ind = Dates.value(df22[i,5] - dinit1) + 1
                        D_obs[ind] = D_obs[ind] + 1
                    end
                end
            end
        end
    end

    println("Mortes: ", sum(D_obs))
    println("Número de dias: ", n_days)
    return D_obs
end

function plota_dados_20()
    D_obs = le_SIVEP_Gripe("INFLUD20-30-05-2022.csv", 0)
    println(D_obs)

    figure(1)
    plot(1:length(D_obs), D_obs, label = "Mortes")
    title("Mortes SP capital - SIVEP-Gripe 2020")
    xlabel("Dia (a partir de 26/02/2020)")
    ylabel("Número de indivíduos")
    legend()
    savefig("mortes_SP_2020.png")
end

function plota_dados_21()
    D_obs = le_SIVEP_Gripe("INFLUD21-30-05-2022.csv", 1)
    println(D_obs)

    figure(1)
    plot(1:length(D_obs), D_obs, label = "Mortes")
    title("Mortes SP capital - SIVEP-Gripe 2021")
    xlabel("Dia (a partir de 01/01/2021)")
    ylabel("Número de indivíduos")
    legend()
    savefig("mortes_SP_2021_v1.png")
end

function plota_dados_22()
    D_obs = le_SIVEP_Gripe("INFLUD22-30-05-2022.csv", 2)
    println(D_obs)

    figure(1)
    plot(1:length(D_obs), D_obs, label = "Mortes")
    title("Mortes SP capital - SIVEP-Gripe 2022")
    xlabel("Dia (a partir de 01/01/2022)")
    ylabel("Número de indivíduos")
    legend()
    savefig("mortes_SP_2022_v1.png")
end

function plota_dados_20_21()
    D_obs = le_SIVEP_Gripe_2("INFLUD20-30-05-2022.csv", "INFLUD21-30-05-2022.csv", 1)
    println(D_obs)

    figure(1)
    plot(1:length(D_obs), D_obs, label = "Mortes")
    title("Mortes SP capital - SIVEP-Gripe 2020/2021")
    xlabel("Dia (a partir de 11/11/2020)")
    ylabel("Número de indivíduos")
    legend()
    savefig("mortes_SP_2021_v2.png")
end

function plota_dados_21_22()
    D_obs = le_SIVEP_Gripe_2("INFLUD21-30-05-2022.csv", "INFLUD22-30-05-2022.csv", 2)
    println(D_obs)

    figure(1)
    plot(1:length(D_obs), D_obs, label = "Mortes")
    title("Mortes SP capital - SIVEP-Gripe 2021/2022")
    xlabel("Dia (a partir de 25/11/2021)")
    ylabel("Número de indivíduos")
    legend()
    savefig("mortes_SP_2022_v2.png")
end

#plota_dados_21_22()