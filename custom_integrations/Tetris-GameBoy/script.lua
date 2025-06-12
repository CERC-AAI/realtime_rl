prev_score = 0

function points()
  reward = data.score_bcd - prev_score
  prev_score = data.score_bcd
  return reward
end
  
    
function done_check()
  if data.game_state == 0 then
    return false
  end
  return true
end 
