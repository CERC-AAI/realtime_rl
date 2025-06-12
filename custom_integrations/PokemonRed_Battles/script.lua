prev_opp_hp1 = 0
prev_opp_hp2 = 0
prev_opp_hp3 = 0
prev_opp_hp4 = 0
prev_opp_hp5 = 0
prev_opp_hp6 = 0
prev_my_hp1 = 0
prev_my_hp2 = 0
prev_my_hp3 = 0
prev_my_hp4 = 0
prev_my_hp5 = 0
prev_my_hp6 = 0
function faints()
  if data.my_hp1 == 0 then
    if prev_my_hp1 > 0 then
      prev_opp_hp1 = data.opp_hp1
      prev_opp_hp2 = data.opp_hp2
      prev_opp_hp3 = data.opp_hp3
      prev_opp_hp4 = data.opp_hp4
      prev_opp_hp5 = data.opp_hp5
      prev_opp_hp6 = data.opp_hp6
      prev_my_hp1 = data.my_hp1
      prev_my_hp2 = data.my_hp2
      prev_my_hp3 = data.my_hp3
      prev_my_hp4 = data.my_hp4
      prev_my_hp5 = data.my_hp5
      prev_my_hp6 = data.my_hp6
      return -1
    end
  end
  if data.my_hp2 == 0 then
    if prev_my_hp2 > 0 then
      prev_opp_hp1 = data.opp_hp1
      prev_opp_hp2 = data.opp_hp2
      prev_opp_hp3 = data.opp_hp3
      prev_opp_hp4 = data.opp_hp4
      prev_opp_hp5 = data.opp_hp5
      prev_opp_hp6 = data.opp_hp6
      prev_my_hp1 = data.my_hp1
      prev_my_hp2 = data.my_hp2
      prev_my_hp3 = data.my_hp3
      prev_my_hp4 = data.my_hp4
      prev_my_hp5 = data.my_hp5
      prev_my_hp6 = data.my_hp6
      return -1
    end
  end
  if data.my_hp3 == 0 then
    if prev_my_hp3 > 0 then
      prev_opp_hp1 = data.opp_hp1
      prev_opp_hp2 = data.opp_hp2
      prev_opp_hp3 = data.opp_hp3
      prev_opp_hp4 = data.opp_hp4
      prev_opp_hp5 = data.opp_hp5
      prev_opp_hp6 = data.opp_hp6
      prev_my_hp1 = data.my_hp1
      prev_my_hp2 = data.my_hp2
      prev_my_hp3 = data.my_hp3
      prev_my_hp4 = data.my_hp4
      prev_my_hp5 = data.my_hp5
      prev_my_hp6 = data.my_hp6
      return -1
    end
  end
  if data.my_hp4 == 0 then
    if prev_my_hp4 > 0 then
      prev_opp_hp1 = data.opp_hp1
      prev_opp_hp2 = data.opp_hp2
      prev_opp_hp3 = data.opp_hp3
      prev_opp_hp4 = data.opp_hp4
      prev_opp_hp5 = data.opp_hp5
      prev_opp_hp6 = data.opp_hp6
      prev_my_hp1 = data.my_hp1
      prev_my_hp2 = data.my_hp2
      prev_my_hp3 = data.my_hp3
      prev_my_hp4 = data.my_hp4
      prev_my_hp5 = data.my_hp5
      prev_my_hp6 = data.my_hp6
      return -1
    end
  end
  if data.my_hp5 == 0 then
    if prev_my_hp5 > 0 then
      prev_opp_hp1 = data.opp_hp1
      prev_opp_hp2 = data.opp_hp2
      prev_opp_hp3 = data.opp_hp3
      prev_opp_hp4 = data.opp_hp4
      prev_opp_hp5 = data.opp_hp5
      prev_opp_hp6 = data.opp_hp6
      prev_my_hp1 = data.my_hp1
      prev_my_hp2 = data.my_hp2
      prev_my_hp3 = data.my_hp3
      prev_my_hp4 = data.my_hp4
      prev_my_hp5 = data.my_hp5
      prev_my_hp6 = data.my_hp6
      return -1
    end
  end
  if data.my_hp6 == 0 then
    if prev_my_hp6 > 0 then
      prev_opp_hp1 = data.opp_hp1
      prev_opp_hp2 = data.opp_hp2
      prev_opp_hp3 = data.opp_hp3
      prev_opp_hp4 = data.opp_hp4
      prev_opp_hp5 = data.opp_hp5
      prev_opp_hp6 = data.opp_hp6
      prev_my_hp1 = data.my_hp1
      prev_my_hp2 = data.my_hp2
      prev_my_hp3 = data.my_hp3
      prev_my_hp4 = data.my_hp4
      prev_my_hp5 = data.my_hp5
      prev_my_hp6 = data.my_hp6
      return -1
    end
  end
  if data.opp_hp1 == 0 then
    if prev_opp_hp1 > 0 then
      prev_opp_hp1 = data.opp_hp1
      prev_opp_hp2 = data.opp_hp2
      prev_opp_hp3 = data.opp_hp3
      prev_opp_hp4 = data.opp_hp4
      prev_opp_hp5 = data.opp_hp5
      prev_opp_hp6 = data.opp_hp6
      prev_my_hp1 = data.my_hp1
      prev_my_hp2 = data.my_hp2
      prev_my_hp3 = data.my_hp3
      prev_my_hp4 = data.my_hp4
      prev_my_hp5 = data.my_hp5
      prev_my_hp6 = data.my_hp6
      return 1
    end
  end
  if data.opp_hp2 == 0 then
    if prev_opp_hp2 > 0 then
      prev_opp_hp1 = data.opp_hp1
      prev_opp_hp2 = data.opp_hp2
      prev_opp_hp3 = data.opp_hp3
      prev_opp_hp4 = data.opp_hp4
      prev_opp_hp5 = data.opp_hp5
      prev_opp_hp6 = data.opp_hp6
      prev_my_hp1 = data.my_hp1
      prev_my_hp2 = data.my_hp2
      prev_my_hp3 = data.my_hp3
      prev_my_hp4 = data.my_hp4
      prev_my_hp5 = data.my_hp5
      prev_my_hp6 = data.my_hp6
      return 1
    end
  end
  if data.opp_hp3 == 0 then
    if prev_opp_hp3 > 0 then
      prev_opp_hp1 = data.opp_hp1
      prev_opp_hp2 = data.opp_hp2
      prev_opp_hp3 = data.opp_hp3
      prev_opp_hp4 = data.opp_hp4
      prev_opp_hp5 = data.opp_hp5
      prev_opp_hp6 = data.opp_hp6
      prev_my_hp1 = data.my_hp1
      prev_my_hp2 = data.my_hp2
      prev_my_hp3 = data.my_hp3
      prev_my_hp4 = data.my_hp4
      prev_my_hp5 = data.my_hp5
      prev_my_hp6 = data.my_hp6
      return 1
    end
  end
  if data.opp_hp4 == 0 then
    if prev_opp_hp4 > 0 then
      prev_opp_hp1 = data.opp_hp1
      prev_opp_hp2 = data.opp_hp2
      prev_opp_hp3 = data.opp_hp3
      prev_opp_hp4 = data.opp_hp4
      prev_opp_hp5 = data.opp_hp5
      prev_opp_hp6 = data.opp_hp6
      prev_my_hp1 = data.my_hp1
      prev_my_hp2 = data.my_hp2
      prev_my_hp3 = data.my_hp3
      prev_my_hp4 = data.my_hp4
      prev_my_hp5 = data.my_hp5
      prev_my_hp6 = data.my_hp6
      return 1
    end
  end
  if data.opp_hp5 == 0 then
    if prev_opp_hp5 > 0 then
      prev_opp_hp1 = data.opp_hp1
      prev_opp_hp2 = data.opp_hp2
      prev_opp_hp3 = data.opp_hp3
      prev_opp_hp4 = data.opp_hp4
      prev_opp_hp5 = data.opp_hp5
      prev_opp_hp6 = data.opp_hp6
      prev_my_hp1 = data.my_hp1
      prev_my_hp2 = data.my_hp2
      prev_my_hp3 = data.my_hp3
      prev_my_hp4 = data.my_hp4
      prev_my_hp5 = data.my_hp5
      prev_my_hp6 = data.my_hp6
      return 1
    end
  end
  if data.opp_hp6 == 0 then
    if prev_opp_hp6 > 0 then
      prev_opp_hp1 = data.opp_hp1
      prev_opp_hp2 = data.opp_hp2
      prev_opp_hp3 = data.opp_hp3
      prev_opp_hp4 = data.opp_hp4
      prev_opp_hp5 = data.opp_hp5
      prev_opp_hp6 = data.opp_hp6
      prev_my_hp1 = data.my_hp1
      prev_my_hp2 = data.my_hp2
      prev_my_hp3 = data.my_hp3
      prev_my_hp4 = data.my_hp4
      prev_my_hp5 = data.my_hp5
      prev_my_hp6 = data.my_hp6
      return 1
    end
  end
  prev_opp_hp1 = data.opp_hp1
  prev_opp_hp2 = data.opp_hp2
  prev_opp_hp3 = data.opp_hp3
  prev_opp_hp4 = data.opp_hp4
  prev_opp_hp5 = data.opp_hp5
  prev_opp_hp6 = data.opp_hp6
  prev_my_hp1 = data.my_hp1
  prev_my_hp2 = data.my_hp2
  prev_my_hp3 = data.my_hp3
  prev_my_hp4 = data.my_hp4
  prev_my_hp5 = data.my_hp5
  prev_my_hp6 = data.my_hp6
  return 0
end
  
    
function done_check()
  if data.my_hp1 == 0 then
    if data.my_hp2 == 0 then
      if data.my_hp3 == 0 then
        if data.my_hp4 == 0 then
          if data.my_hp5 == 0 then
            if data.my_hp6 == 0 then
              return true
            end
          end
        end
      end 
    end 
  end
  if data.opp_hp1 == 0 then
    if data.opp_hp2 == 0 then
      if data.opp_hp3 == 0 then
        if data.opp_hp4 == 0 then
          if data.opp_hp5 == 0 then
            if data.opp_hp6 == 0 then
              return true
            end
          end
        end
      end 
    end 
  end
  return false
end
