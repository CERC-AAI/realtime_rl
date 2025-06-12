prev_owned1 = -1000
prev_owned2 = -1000
prev_owned3 = -1000
prev_owned4 = -1000
prev_owned5 = -1000
prev_owned6 = -1000
prev_owned7 = -1000
prev_owned8 = -1000
prev_owned9 = -1000
prev_owned10 = -1000
prev_owned11 = -1000
prev_owned12 = -1000
prev_owned13 = -1000
prev_owned14 = -1000
prev_owned15 = -1000
prev_owned16 = -1000
prev_owned17 = -1000
prev_owned18 = -1000
prev_owned19 = -1000
old_battle = 0
over = 0 
catch = 0
prev_box_qnt = -10

function caught()
  if (data.box_qnt - prev_box_qnt) == 1 then
    rew_received = 1 - catch
    catch = 1
    prev_owned1 = data.Owned1
    prev_owned2 = data.Owned2
    prev_owned3 = data.Owned3
    prev_owned4 = data.Owned4
    prev_owned5 = data.Owned5
    prev_owned6 = data.Owned6
    prev_owned7 = data.Owned7
    prev_owned8 = data.Owned8
    prev_owned9 = data.Owned9
    prev_owned10 = data.Owned10
    prev_owned11 = data.Owned11
    prev_owned12 = data.Owned12
    prev_owned13 = data.Owned13
    prev_owned14 = data.Owned14
    prev_owned15 = data.Owned15
    prev_owned16 = data.Owned16
    prev_owned17 = data.Owned17
    prev_owned18 = data.Owned18
    prev_owned19 = data.Owned19
    prev_box_qnt = data.box_qnt
    return rew_received
  end 
  if data.Owned1 ~= prev_owned1 then
    if (prev_owned1 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned2 ~= prev_owned2 then
    if (prev_owned2 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned3 ~= prev_owned3 then
    if (prev_owned3 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned4 ~= prev_owned4 then
    if (prev_owned4 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned5 ~= prev_owned5 then
    if (prev_owned5 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned6 ~= prev_owned6 then
    if (prev_owned6 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned7 ~= prev_owned7 then
    if (prev_owned7 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned8 ~= prev_owned8 then
    if (prev_owned8 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned9 ~= prev_owned9 then
    if (prev_owned9 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned10 ~= prev_owned10 then
    if (prev_owned10 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned11 ~= prev_owned11 then
    if (prev_owned11 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned12 ~= prev_owned12 then
    if (prev_owned12 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned13 ~= prev_owned13 then
    if (prev_owned13 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned14 ~= prev_owned14 then
    if (prev_owned14 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned15 ~= prev_owned15 then
    if (prev_owned15 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned16 ~= prev_owned16 then
    if (prev_owned16 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned17 ~= prev_owned17 then
    if (prev_owned17 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned18 ~= prev_owned18 then
    if (prev_owned18 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.Owned19 ~= prev_owned19 then
    if (prev_owned19 ~= -1000 and catch == 0) then
      catch = 1
      prev_owned1 = data.Owned1
      prev_owned2 = data.Owned2
      prev_owned3 = data.Owned3
      prev_owned4 = data.Owned4
      prev_owned5 = data.Owned5
      prev_owned6 = data.Owned6
      prev_owned7 = data.Owned7
      prev_owned8 = data.Owned8
      prev_owned9 = data.Owned9
      prev_owned10 = data.Owned10
      prev_owned11 = data.Owned11
      prev_owned12 = data.Owned12
      prev_owned13 = data.Owned13
      prev_owned14 = data.Owned14
      prev_owned15 = data.Owned15
      prev_owned16 = data.Owned16
      prev_owned17 = data.Owned17
      prev_owned18 = data.Owned18
      prev_owned19 = data.Owned19
      prev_box_qnt = data.box_qnt
      return 1
    end
  end
  if data.InBattle < old_battle then
    old_battle = data.InBattle 
    return catch - 1
  end 
  prev_owned1 = data.Owned1
  prev_owned2 = data.Owned2
  prev_owned3 = data.Owned3
  prev_owned4 = data.Owned4
  prev_owned5 = data.Owned5
  prev_owned6 = data.Owned6
  prev_owned7 = data.Owned7
  prev_owned8 = data.Owned8
  prev_owned9 = data.Owned9
  prev_owned10 = data.Owned10
  prev_owned11 = data.Owned11
  prev_owned12 = data.Owned12
  prev_owned13 = data.Owned13
  prev_owned14 = data.Owned14
  prev_owned15 = data.Owned15
  prev_owned16 = data.Owned16
  prev_owned17 = data.Owned17
  prev_owned18 = data.Owned18
  prev_owned19 = data.Owned19
  old_battle = data.InBattle
  prev_box_qnt = data.box_qnt
  return 0
end
  
    
function done_check()
  if data.InBattle == 0 then
    return true
  end
  return false
end
