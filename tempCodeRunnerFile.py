Rkp = np.zeros((param.nbVarF - 1, param.nbVarF - 1))
        # Rkp[:3, :3] = np.identity(3) #Translation constraint
        # Rkp[3:, 3:] = q2R(param.Mu[-4:, i]) #Orientation constraint