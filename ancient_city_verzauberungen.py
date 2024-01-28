import random


def buch_lotterie(virtuelle_kisten):
    zaehler = 0

    for _ in range(virtuelle_kisten):
        anzahl_kisten_slots_mit_loot = random.randint(5, 10)
        buecher = []

        for _ in range(anzahl_kisten_slots_mit_loot):
            outcomes = ["Ent_AlleAnderen", "Ent_SwiftSneak", "WasAnderes"]
            probabilities = [5/86, 3/86, 78/86]

            # Draw one outcome based on the specified probabilities
            slot_inhalt = random.choices(outcomes, weights=probabilities, k=1)[0]

            if slot_inhalt == "Ent_AlleAnderen": # Enchantment Buch (der Eintrag ohne SwiftSneak)
                verzauberung = random.randint(1, 37)  # Ziehe zufällige Verzauberung
                if verzauberung == 1:
                    verzauberung_level = random.choice(["Mending"])
                elif verzauberung == 2:
                    verzauberung_level = random.choice(["Unbreaking1", "Unbreaking2", "Unbreaking3"])
                elif verzauberung == 3:
                    verzauberung_level = random.choice(["Curse of Vanishing"])
                elif verzauberung == 4:
                    verzauberung_level = random.choice(["Aqua Affinity"])
                elif verzauberung == 5:
                    verzauberung_level = random.choice(["Blast Protection1", "Blast Protection2", "Blast Protection3", "Blast Protection4"])
                elif verzauberung == 6:
                    verzauberung_level = random.choice(["Curse of Binding"])
                elif verzauberung == 7:
                    verzauberung_level = random.choice(["Depth Strider1", "Depth Strider2", "Depth Strider3"])
                elif verzauberung == 8:
                    verzauberung_level = random.choice(["Feather Falling1", "Feather Falling2", "Feather Falling3", "Feather Falling4"])
                elif verzauberung == 9:
                    verzauberung_level = random.choice(["Fire Protection1", "Fire Protection2", "Fire Protection3", "Fire Protection4"])
                elif verzauberung == 10:
                    verzauberung_level = random.choice(["Frost Walker1", "Frost Walker2"])
                elif verzauberung == 11:
                    verzauberung_level = random.choice(["Projectile Protection1", "Projectile Protection2", "Projectile Protection3", "Projectile Protection4"])
                elif verzauberung == 12:
                    verzauberung_level = random.choice(["Protection1", "Protection2", "Protection3", "Protection4"])
                elif verzauberung == 13:
                    verzauberung_level = random.choice(["Respiration1", "Respiration2", "Respiration3"])
                elif verzauberung == 14:
                    verzauberung_level = random.choice(["Thorns1", "Thorns2", "Thorns3"])
                elif verzauberung == 15:
                    verzauberung_level = random.choice(["Bane of Arthropods1", "Bane of Arthropods2", "Bane of Arthropods3", "Bane of Arthropods4", "Bane of Arthropods5"])
                elif verzauberung == 16:
                    verzauberung_level = random.choice(["Efficiency1", "Efficiency2", "Efficiency3", "Efficiency4", "Efficiency5"])
                elif verzauberung == 17:
                    verzauberung_level = random.choice(["Fire Aspect1", "Fire Aspect2"])
                elif verzauberung == 18:
                    verzauberung_level = random.choice(["Looting1", "Looting2", "Looting3"])
                elif verzauberung == 19:
                    verzauberung_level = random.choice(["Impaling1", "Impaling2", "Impaling3", "Impaling4", "Impaling5"])
                elif verzauberung == 20:
                    verzauberung_level = random.choice(["Knockback1", "Knockback2"])
                elif verzauberung == 21:
                    verzauberung_level = random.choice(["Sharpness1", "Sharpness2", "Sharpness3", "Sharpness4", "Sharpness5"])
                elif verzauberung == 22:
                    verzauberung_level = random.choice(["Smite1", "Smite2", "Smite3", "Smite4", "Smite5"])
                elif verzauberung == 23:
                    verzauberung_level = random.choice(["Sweeping Edge1", "Sweeping Edge2", "Sweeping Edge3"])
                elif verzauberung == 24:
                    verzauberung_level = random.choice(["Channeling"])
                elif verzauberung == 25:
                    verzauberung_level = random.choice(["Flame"])
                elif verzauberung == 26:
                    verzauberung_level = random.choice(["Impaling1", "Impaling2", "Impaling3", "Impaling4", "Impaling5"])
                elif verzauberung == 27:
                    verzauberung_level = random.choice(["Infinity"])
                elif verzauberung == 28:
                    verzauberung_level = random.choice(["Loyalty1", "Loyalty2", "Loyalty3"])
                elif verzauberung == 29:
                    verzauberung_level = random.choice(["Riptide1", "Riptide2", "Riptide3"])
                elif verzauberung == 30:
                    verzauberung_level = random.choice(["Piercing1", "Piercing2", "Piercing3", "Piercing4"])
                elif verzauberung == 31:
                    verzauberung_level = random.choice(["Power1", "Power2", "Power3", "Power4", "Power5"])
                elif verzauberung == 32:
                    verzauberung_level = random.choice(["Punch1", "Punch2"])
                elif verzauberung == 33:
                    verzauberung_level = random.choice(["Quick Charge1", "Quick Charge2", "Quick Charge3"])
                elif verzauberung == 34:
                    verzauberung_level = random.choice(["Fortune1", "Fortune2", "Fortune3"])
                elif verzauberung == 35:
                    verzauberung_level = random.choice(["Luck of the Sea1", "Luck of the Sea2", "Luck of the Sea3"])
                elif verzauberung == 36:
                    verzauberung_level = random.choice(["Lure1", "Lure2", "Lure3"])
                elif verzauberung == 37:
                    verzauberung_level = random.choice(["Silk Touch"])

                buecher.append(verzauberung_level)

            if slot_inhalt == "Ent_SwiftSneak": # Enchantment Buch (der Eintrag für SwiftSneak)
                verzauberung_level = random.choice(["SwiftSneak1", "SwiftSneak2", "SwiftSneak3"])
                buecher.append(verzauberung_level)

        if any(buecher.count(verzauberung_level) == 2 for verzauberung_level in buecher):
            zaehler += 1

    overall_probability = (zaehler / virtuelle_kisten) * 100
    print(f"Zwei gleiche Verzauberungen: {overall_probability:.4f}%")
    return overall_probability

virtuelle_kisten = 10_000_0000
wahrscheinlichkeit_2_gleiche_enchantments = buch_lotterie(virtuelle_kisten)
