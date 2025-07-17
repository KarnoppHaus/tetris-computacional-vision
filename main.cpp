#include <raylib.h>
#include "game.h"
#include "colors.h"
#include <iostream>

double lastUpdateTime = 0;

bool EventTriggered(double interval)
{
    double currentTime = GetTime() / 3;
    if (currentTime - lastUpdateTime >= interval)
    {
        lastUpdateTime = currentTime;
        return true;
    }
    return false;
}

int main()
{
    InitWindow(500, 620, "Tetris Computacional Vision");
    SetTargetFPS(60);

    Font font = LoadFontEx("monogram.ttf", 64, 0, 0);

    // SetTargetFPS(30);

    Game game = Game();

    while (WindowShouldClose() == false)
    {
        if (game.inMenu == true) {
            BeginDrawing();
            ClearBackground(darkBlue);
            DrawTextEx(font, "TETRIS",        {200,  35}, 38, 2, WHITE);
            DrawTextEx(font, "COMPUTACIONAL", {150,  80}, 38, 2, WHITE);
            DrawTextEx(font, "VISION",        {200, 125}, 38, 2, WHITE);
            DrawRectangleRounded({80, 300, 340, 50}, 0.2, 10, lightBlue);
            DrawTextEx(font, "'ROTACIONE' PARA INICIAR", {85, 308}, 25, 2, WHITE);
            game.HandleInput();
            EndDrawing();
        } 
        
        else if (game.inMenu == false) {
            UpdateMusicStream(game.music);
            game.HandleInput();
            if (EventTriggered(0.2))
            {
                game.MoveBlockDown();
            }

            BeginDrawing();
            ClearBackground(darkBlue);
            DrawTextEx(font, "Score", {365, 15}, 38, 2, WHITE);
            DrawTextEx(font, "Next", {370, 175}, 38, 2, WHITE);
            if (game.gameOver)
            {
                DrawTextEx(font, "GAME OVER", {320, 450}, 38, 2, WHITE);
            }
            DrawRectangleRounded({320, 55, 170, 60}, 0.3, 6, lightBlue);

            char scoreText[10];
            sprintf(scoreText, "%d", game.score);
            Vector2 textSize = MeasureTextEx(font, scoreText, 38, 2);

            DrawTextEx(font, scoreText, {320 + (170 - textSize.x) / 2, 65}, 38, 2, WHITE);
            DrawRectangleRounded({320, 215, 170, 180}, 0.3, 6, lightBlue);
            game.Draw();
            EndDrawing();
        }
    }

    CloseWindow();
}