#include "BaseCar.h"
#include <cmath>

namespace physics {

BaseCar::BaseCar(float width,float length)
{
    m_Width = width;
    m_Length = length;
    m_Body = 0;
}

BaseCar::~BaseCar()
{
}

b2Vec2 BaseCar::GetPosition()
{
    // ASSERT (m_Body!=0)
    return m_Body->GetPosition();
}

float BaseCar::GetAngle()
{ 
    return m_Body->GetAngle();
}

void BaseCar::SetPosition(b2Vec2 pos)
{
    float angle = m_Body->GetAngle();
    m_Body->SetTransform(pos,angle);
}

void BaseCar::SetAngle(float angle)
{
    b2Vec2 pos = m_Body->GetPosition();
    m_Body->SetTransform(pos,angle);

}

void BaseCar::SetSpeed(b2Vec2 speed)
{
    m_Body->SetLinearVelocity(speed);
}

float BaseCar::GetSpeed() 
{
    b2Vec2 vel = m_Body->GetLinearVelocity();
    float speed = std::sqrt(vel.x * vel.x + vel.y * vel.y);
    return speed;
}

}   // namespace physics

